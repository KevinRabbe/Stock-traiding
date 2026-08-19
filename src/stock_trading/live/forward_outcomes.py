from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.core import Source, as_utc
from stock_trading.local_secrets import load_tiingo_credentials
from stock_trading.market import (
    DuckDbMarketStore,
    TiingoClient,
    TiingoNormalizer,
    build_standard_labels,
)
from stock_trading.storage import FileRawStore

from .run_current_paper_shadow import _load_runtime_config
from .session_calendar import XnysExecutionSessionResolver


_FORWARD_HORIZONS = (5, 20, 60)


@dataclass(frozen=True, slots=True)
class ForwardDecisionInstance:
    batch_id: str
    candidate_id: str
    company_id: str
    security_id: str
    execution_date: date
    strategy_decisions: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class KnownSeriesSync:
    security_id: str
    ticker: str
    start_date: date | None
    requested_end_date: date
    coverage_end_date: date | None
    downloaded: bool
    bars_added: int

    @property
    def complete(self) -> bool:
        return self.coverage_end_date is not None and self.coverage_end_date >= self.requested_end_date


def refresh_forward_outcome_scorecard(
    *,
    data_root: str | Path = "data",
    runtime_dir: str | Path = "data/runtime",
    as_of: datetime | None = None,
) -> dict:
    """Materialize realized labels for every persisted forward decision candidate.

    Decision diagnostics are immutable evidence of what each strategy knew and chose
    at decision time. This routine never re-scores a model and has no trading
    authority. It only extends already-known market series through the last completed
    XNYS session and joins matured 5/20/60-session labels to those immutable decisions.

    The observation identity is ``batch_id + candidate_id`` rather than candidate ID
    alone. Multiple filing batches for one company and execution session can therefore
    remain distinct forward information sets even though the strategy candidate ID is
    intentionally company/session stable.
    """

    data_root = Path(data_root)
    runtime_dir = Path(runtime_dir)
    cutoff = as_utc(as_of or datetime.now(timezone.utc))
    resolver = XnysExecutionSessionResolver()
    completed_session = resolver.last_completed_session(cutoff)
    config = _load_runtime_config(runtime_dir)
    benchmark_security_id = str(config["benchmark_security_id"])
    market_store = DuckDbMarketStore(Path(str(config["market_db"])))
    market_store.enable_read_cache(max_series=64)

    instances = _load_forward_decision_instances(runtime_dir / "decision_diagnostics")
    tracked_securities = sorted(
        {
            (item.security_id, _ticker_for_security(market_store, item.security_id, completed_session))
            for item in instances
            if item.execution_date <= completed_session
        }
    )

    needs_sync: list[tuple[str, str]] = []
    for security_id, ticker in tracked_securities:
        bounds = market_store.date_bounds(security_id, ticker)
        if bounds is None or bounds[1] < completed_session:
            needs_sync.append((security_id, ticker))
    benchmark_bounds = market_store.date_bounds(benchmark_security_id, "SPY")
    benchmark_needs_sync = benchmark_bounds is None or benchmark_bounds[1] < completed_session

    sync_results: list[KnownSeriesSync] = []
    benchmark_sync: KnownSeriesSync | None = None
    if needs_sync or benchmark_needs_sync:
        raw_store = FileRawStore(data_root / "raw")
        credentials = load_tiingo_credentials(data_root)
        with TiingoClient(credentials.token) as client:
            for security_id, ticker in needs_sync:
                sync_results.append(
                    _sync_known_series(
                        client,
                        raw_store=raw_store,
                        market_store=market_store,
                        security_id=security_id,
                        ticker=ticker,
                        sync_end_date=completed_session,
                    )
                )
            if benchmark_needs_sync:
                benchmark_sync = _sync_known_series(
                    client,
                    raw_store=raw_store,
                    market_store=market_store,
                    security_id=benchmark_security_id,
                    ticker="SPY",
                    sync_end_date=completed_session,
                )
        market_store.clear_read_cache()

    observations: list[dict] = []
    matured_counts = {str(horizon): 0 for horizon in _FORWARD_HORIZONS}
    emitted_strategy_decisions = 0
    rejected_strategy_decisions = 0
    strategy_decision_count = 0
    for instance in instances:
        labels = _realized_labels(
            market_store,
            instance,
            benchmark_security_id=benchmark_security_id,
            completed_session=completed_session,
        )
        label_payload: dict[str, dict] = {}
        for label in labels:
            matured_counts[str(label.horizon)] += 1
            label_payload[str(label.horizon)] = {
                "horizon_sessions": label.horizon,
                "start_date": label.start_date.isoformat(),
                "end_date": label.end_date.isoformat(),
                "stock_return": label.stock_return,
                "benchmark_return": label.benchmark_return,
                "alpha": label.alpha,
                "max_favorable_excursion": label.max_favorable_excursion,
                "max_adverse_excursion": label.max_adverse_excursion,
            }

        decisions: list[dict] = []
        for decision in instance.strategy_decisions:
            strategy_decision_count += 1
            emitted = bool(decision["emitted"])
            if emitted:
                emitted_strategy_decisions += 1
            else:
                rejected_strategy_decisions += 1
            decisions.append(decision)

        observations.append(
            {
                "observation_id": f"{instance.batch_id}:{instance.candidate_id}",
                "batch_id": instance.batch_id,
                "candidate_id": instance.candidate_id,
                "company_id": instance.company_id,
                "security_id": instance.security_id,
                "execution_date": instance.execution_date.isoformat(),
                "strategy_decisions": decisions,
                "realized_labels": label_payload,
                "matured_horizon_count": len(label_payload),
                "fully_matured": len(label_payload) == len(_FORWARD_HORIZONS),
            }
        )

    fully_matured = sum(1 for item in observations if item["fully_matured"])
    total_possible_labels = len(observations) * len(_FORWARD_HORIZONS)
    matured_total = sum(matured_counts.values())
    lagging = [item for item in sync_results if not item.complete]
    if benchmark_sync is not None and not benchmark_sync.complete:
        lagging.append(benchmark_sync)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": cutoff.isoformat(),
        "last_completed_xnys_session": completed_session.isoformat(),
        "summary": {
            "diagnostic_batch_count": len({item.batch_id for item in instances}),
            "candidate_decision_instance_count": len(instances),
            "strategy_decision_count": strategy_decision_count,
            "emitted_strategy_decision_count": emitted_strategy_decisions,
            "rejected_strategy_decision_count": rejected_strategy_decisions,
            "matured_horizon_label_count": matured_total,
            "pending_horizon_label_count": max(0, total_possible_labels - matured_total),
            "fully_matured_candidate_instance_count": fully_matured,
            "matured_by_horizon": matured_counts,
        },
        "market_sync": {
            "tracked_security_count": len(tracked_securities),
            "downloaded_series_count": sum(1 for item in sync_results if item.downloaded),
            "bars_added": sum(item.bars_added for item in sync_results),
            "benchmark_downloaded": bool(benchmark_sync and benchmark_sync.downloaded),
            "benchmark_bars_added": benchmark_sync.bars_added if benchmark_sync else 0,
            "lagging_series_count": len(lagging),
            "lagging_series": [_sync_payload(item) for item in lagging],
        },
        "observations": observations,
    }
    path = runtime_dir / "forward_scorecard.json"
    _atomic_json_write(path, payload)
    return {
        "status": "completed",
        "scorecard_path": str(path),
        **payload["summary"],
        "market_sync": payload["market_sync"],
    }


def _load_forward_decision_instances(root: Path) -> tuple[ForwardDecisionInstance, ...]:
    if not root.exists():
        return ()

    result: list[ForwardDecisionInstance] = []
    for path in sorted(root.glob("batch_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid forward decision diagnostic: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"unsupported forward decision diagnostic: {path}")
        batch_id = str(payload.get("batch_id") or "")
        if batch_id != path.stem:
            raise ValueError(f"forward diagnostic batch identity mismatch: {path}")
        strategies = payload.get("strategies")
        if not isinstance(strategies, list) or not strategies:
            raise ValueError(f"forward diagnostic has no strategies: {path}")

        identities: dict[str, tuple[str, str, date]] | None = None
        decisions_by_candidate: dict[str, list[dict]] = {}
        for strategy in strategies:
            if not isinstance(strategy, dict):
                raise ValueError(f"invalid strategy diagnostic in {path}")
            strategy_id = str(strategy.get("strategy_id") or "")
            decisions = strategy.get("decisions")
            if not strategy_id or not isinstance(decisions, list):
                raise ValueError(f"invalid strategy diagnostic in {path}")
            current: dict[str, tuple[str, str, date]] = {}
            for decision in decisions:
                if not isinstance(decision, dict):
                    raise ValueError(f"invalid candidate diagnostic in {path}")
                candidate_id = str(decision.get("candidate_id") or "")
                company_id = str(decision.get("company_id") or "")
                security_id = str(decision.get("security_id") or "")
                try:
                    execution_date = date.fromisoformat(str(decision["execution_date"]))
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"invalid candidate execution date in {path}") from exc
                if not candidate_id or not company_id or not security_id:
                    raise ValueError(f"incomplete candidate identity in {path}")
                current[candidate_id] = (company_id, security_id, execution_date)
                decisions_by_candidate.setdefault(candidate_id, []).append(
                    _compact_strategy_decision(strategy_id, decision)
                )
            if identities is None:
                identities = current
            elif identities != current:
                raise ValueError(
                    f"strategy cohort saw different candidate identities in {path}"
                )

        assert identities is not None
        strategy_count = len(strategies)
        for candidate_id, (company_id, security_id, execution_date) in sorted(identities.items()):
            decisions = decisions_by_candidate.get(candidate_id, [])
            if len(decisions) != strategy_count:
                raise ValueError(f"candidate strategy decision count differs in {path}")
            result.append(
                ForwardDecisionInstance(
                    batch_id=batch_id,
                    candidate_id=candidate_id,
                    company_id=company_id,
                    security_id=security_id,
                    execution_date=execution_date,
                    strategy_decisions=tuple(sorted(decisions, key=lambda item: item["strategy_id"])),
                )
            )
    return tuple(result)


def _compact_strategy_decision(strategy_id: str, decision: dict) -> dict:
    horizons = decision.get("horizons")
    if not isinstance(horizons, list):
        raise ValueError(f"candidate diagnostic for {strategy_id} has no horizons")
    return {
        "strategy_id": strategy_id,
        "emitted": bool(decision.get("emitted")),
        "rejection_reason": str(decision.get("rejection_reason") or ""),
        "chosen_horizon": decision.get("chosen_horizon"),
        "final_percentile": float(decision.get("final_percentile", 0.0)),
        "rank_threshold": float(decision.get("rank_threshold", 0.0)),
        "horizons": [
            {
                "horizon_sessions": int(item["horizon_sessions"]),
                "expected_return": float(item["expected_return"]),
                "expected_alpha": float(item["expected_alpha"]),
                "expected_downside": float(item["expected_downside"]),
                "probability_positive": float(item["probability_positive"]),
                "combined_signal": float(item["combined_signal"]),
                "eligible": bool(item["eligible"]),
            }
            for item in horizons
        ],
    }


def _ticker_for_security(
    market_store: DuckDbMarketStore,
    security_id: str,
    completed_session: date,
) -> str:
    bars = market_store.bars_before(
        security_id,
        completed_session + timedelta(days=1),
        1,
    )
    if not bars:
        raise RuntimeError(f"forward candidate security has no stored market history: {security_id}")
    return bars[-1].ticker


def _sync_known_series(
    client: TiingoClient,
    *,
    raw_store: FileRawStore,
    market_store: DuckDbMarketStore,
    security_id: str,
    ticker: str,
    sync_end_date: date,
) -> KnownSeriesSync:
    bounds = market_store.date_bounds(security_id, ticker)
    if bounds is None:
        raise RuntimeError(f"tracked security has no stored market series: {security_id}/{ticker}")
    if bounds[1] >= sync_end_date:
        return KnownSeriesSync(
            security_id=security_id,
            ticker=ticker,
            start_date=None,
            requested_end_date=sync_end_date,
            coverage_end_date=bounds[1],
            downloaded=False,
            bars_added=0,
        )

    start = bounds[1] + timedelta(days=1)
    raw = client.fetch_prices(ticker, start, sync_end_date)
    raw_store.put(raw)
    bars = TiingoNormalizer().parse_prices(
        raw,
        security_id=security_id,
        ticker=ticker,
    )
    before = market_store.count_bars(security_id, ticker, start, sync_end_date)
    market_store.put_many(bars)
    after = market_store.count_bars(security_id, ticker, start, sync_end_date)
    refreshed = market_store.date_bounds(security_id, ticker)
    return KnownSeriesSync(
        security_id=security_id,
        ticker=ticker,
        start_date=start,
        requested_end_date=sync_end_date,
        coverage_end_date=refreshed[1] if refreshed is not None else None,
        downloaded=True,
        bars_added=max(0, after - before),
    )


def _realized_labels(
    market_store: DuckDbMarketStore,
    instance: ForwardDecisionInstance,
    *,
    benchmark_security_id: str,
    completed_session: date,
):
    if instance.execution_date > completed_session:
        return ()
    stock_bars = tuple(
        bar
        for bar in market_store.bars_from(instance.security_id, instance.execution_date, 80)
        if bar.date <= completed_session
    )
    benchmark_bars = tuple(
        bar
        for bar in market_store.bars_from(benchmark_security_id, instance.execution_date, 80)
        if bar.date <= completed_session
    )
    labels = build_standard_labels(
        stock_bars,
        benchmark_bars,
        horizons=_FORWARD_HORIZONS,
    )
    return tuple(label for label in labels if label.start_date == instance.execution_date)


def _sync_payload(item: KnownSeriesSync) -> dict:
    return {
        "security_id": item.security_id,
        "ticker": item.ticker,
        "start_date": item.start_date.isoformat() if item.start_date else None,
        "requested_end_date": item.requested_end_date.isoformat(),
        "coverage_end_date": (
            item.coverage_end_date.isoformat() if item.coverage_end_date else None
        ),
        "downloaded": item.downloaded,
        "bars_added": item.bars_added,
    }


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
