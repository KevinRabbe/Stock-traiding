from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import httpx

from stock_trading.core import Event, Source, as_utc
from stock_trading.extraction import FileSemanticCache, QwenSemanticExtractor
from stock_trading.lobbying import LdaClient, LdaFilingNormalizer
from stock_trading.market import CandidateSnapshotBuilder, DuckDbMarketStore, TiingoClient, TiingoNormalizer
from stock_trading.market import normalize_company_name
from stock_trading.storage import DuckDbEventStore, FileRawStore

from .candidates import PitCandidateAssembler
from .current_cycle_receipt import batch_id
from .decision_diagnostics import FileStrategyDecisionDiagnosticStore, diagnose_registry
from .forward_outcomes import refresh_forward_outcome_scorecard
from .run_current_paper_shadow import _load_runtime_config
from .runtime_state import load_persisted_shadow_registry
from .runtime_strategy_state import FileRuntimeStrategyStateStore
from .session_calendar import XnysExecutionSessionResolver


@dataclass(frozen=True, order=True, slots=True)
class LdaFilingCursor:
    posted_at: datetime
    filing_uuid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "posted_at", as_utc(self.posted_at))
        if not self.filing_uuid.strip():
            raise ValueError("LDA filing cursor filing_uuid must not be empty")


@dataclass(frozen=True, slots=True)
class LdaShadowPending:
    event_id: str
    company_id: str
    public_time: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_time", as_utc(self.public_time))
        if not self.event_id.strip() or not self.company_id.strip():
            raise ValueError("LDA shadow pending identity must not be empty")


@dataclass(frozen=True, slots=True)
class LdaShadowIntakeState:
    cursor: LdaFilingCursor | None = None
    pending: tuple[LdaShadowPending, ...] = ()
    stale_event_ids: tuple[str, ...] = ()


class FileLdaShadowIntake:
    """Atomic cursor + pending state for measurement-only LDA intake."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> LdaShadowIntakeState:
        if not self.path.exists():
            return LdaShadowIntakeState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid LDA shadow intake state: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported LDA shadow intake schema")
        raw_cursor = payload.get("cursor")
        cursor = None
        if raw_cursor is not None:
            if not isinstance(raw_cursor, dict):
                raise ValueError("invalid LDA shadow cursor")
            cursor = LdaFilingCursor(
                posted_at=datetime.fromisoformat(str(raw_cursor["posted_at"])),
                filing_uuid=str(raw_cursor["filing_uuid"]),
            )
        try:
            pending = tuple(
                LdaShadowPending(
                    event_id=str(item["event_id"]),
                    company_id=str(item["company_id"]),
                    public_time=datetime.fromisoformat(str(item["public_time"])),
                )
                for item in payload.get("pending", ())
            )
            stale = tuple(str(item) for item in payload.get("stale_event_ids", ()))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid LDA shadow intake payload") from exc
        pending_ids = [item.event_id for item in pending]
        if len(pending_ids) != len(set(pending_ids)):
            raise ValueError("duplicate LDA shadow pending event IDs")
        if len(stale) != len(set(stale)):
            raise ValueError("duplicate LDA shadow stale event IDs")
        if set(pending_ids) & set(stale):
            raise ValueError("LDA shadow event cannot be pending and stale")
        return LdaShadowIntakeState(cursor=cursor, pending=pending, stale_event_ids=stale)

    def commit_poll(
        self,
        *,
        cursor: LdaFilingCursor | None,
        events: Iterable[Event],
    ) -> int:
        state = self.load()
        next_cursor = state.cursor
        if cursor is not None and (next_cursor is None or cursor > next_cursor):
            next_cursor = cursor
        pending = {item.event_id: item for item in state.pending}
        stale = set(state.stale_event_ids)
        added = 0
        for event in events:
            if event.source is not Source.LDA or not event.company_id:
                raise ValueError("LDA shadow intake accepts only mapped LDA events")
            item = LdaShadowPending(
                event_id=event.event_id,
                company_id=event.company_id,
                public_time=event.public_time,
            )
            if item.event_id in stale:
                continue
            previous = pending.get(item.event_id)
            if previous is not None and previous != item:
                raise ValueError(f"LDA pending identity changed for {item.event_id}")
            if previous is None:
                pending[item.event_id] = item
                added += 1
        if next_cursor != state.cursor or added:
            self._save(
                LdaShadowIntakeState(
                    cursor=next_cursor,
                    pending=tuple(sorted(pending.values(), key=_pending_sort_key)),
                    stale_event_ids=state.stale_event_ids,
                )
            )
        return added

    def pending(self, *, as_of: datetime | None = None) -> tuple[LdaShadowPending, ...]:
        values = self.load().pending
        if as_of is None:
            return values
        cutoff = as_utc(as_of)
        return tuple(item for item in values if item.public_time <= cutoff)

    def acknowledge(self, event_ids: Iterable[str]) -> int:
        selected = {str(item) for item in event_ids if str(item)}
        if not selected:
            return 0
        state = self.load()
        kept = tuple(item for item in state.pending if item.event_id not in selected)
        removed = len(state.pending) - len(kept)
        if removed:
            self._save(
                LdaShadowIntakeState(
                    cursor=state.cursor,
                    pending=kept,
                    stale_event_ids=state.stale_event_ids,
                )
            )
        return removed

    def dispose_stale(self, event_ids: Iterable[str]) -> int:
        selected = {str(item) for item in event_ids if str(item)}
        if not selected:
            return 0
        state = self.load()
        kept = tuple(item for item in state.pending if item.event_id not in selected)
        removed_ids = {item.event_id for item in state.pending} - {item.event_id for item in kept}
        if not removed_ids:
            return 0
        stale = tuple(sorted(set(state.stale_event_ids) | removed_ids))
        self._save(
            LdaShadowIntakeState(
                cursor=state.cursor,
                pending=kept,
                stale_event_ids=stale,
            )
        )
        return len(removed_ids)

    def _save(self, state: LdaShadowIntakeState) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "cursor": (
                {
                    "posted_at": state.cursor.posted_at.isoformat(),
                    "filing_uuid": state.cursor.filing_uuid,
                }
                if state.cursor is not None
                else None
            ),
            "pending": [
                {
                    "event_id": item.event_id,
                    "company_id": item.company_id,
                    "public_time": item.public_time.isoformat(),
                }
                for item in sorted(state.pending, key=_pending_sort_key)
            ],
            "stale_event_ids": list(state.stale_event_ids),
        }
        _atomic_json_write(self.path, payload)


@dataclass(frozen=True, slots=True)
class LdaShadowPollResult:
    pages_fetched: int
    filings_seen: int
    new_filings_seen: int
    mapped_event_count: int
    semantic_enriched_event_count: int
    unmapped_filing_count: int
    nonmodeled_filing_count: int
    pending_events_added: int
    pending_event_count: int
    cursor: LdaFilingCursor | None


@dataclass(frozen=True, slots=True)
class LdaShadowBatchSelection:
    target_execution_date: date
    selected_event_ids: tuple[str, ...]
    stale_event_ids: tuple[str, ...]
    future_event_ids: tuple[str, ...]


def poll_current_lda_shadow(
    *,
    data_root: str | Path,
    experiment_dir: str | Path,
    runtime_dir: str | Path,
    lda_client: LdaClient,
    extractor: QwenSemanticExtractor,
    as_of: datetime,
    initial_lookback_days: int = 2,
    max_pages: int | None = None,
) -> LdaShadowPollResult:
    """Poll current LDA filings into isolated SHADOW storage with semantic parity."""

    if initial_lookback_days <= 0:
        raise ValueError("initial_lookback_days must be > 0")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be > 0")

    data_root = Path(data_root)
    experiment_dir = Path(experiment_dir)
    runtime_root = Path(runtime_dir)
    shadow_root = runtime_root / "lda_shadow"
    cutoff = as_utc(as_of)
    intake = FileLdaShadowIntake(shadow_root / "intake.json")
    state = intake.load()
    previous_cursor = state.cursor
    posted_after = (
        previous_cursor.posted_at.date() - timedelta(days=1)
        if previous_cursor is not None
        else cutoff.date() - timedelta(days=initial_lookback_days)
    )
    posted_before = cutoff.date()

    modeled_ids = _modeled_company_ids(experiment_dir / "training_rows.jsonl")
    name_index = _modeled_company_name_index(
        data_root / "manifests" / "sec_companies.jsonl",
        modeled_ids,
    )
    aliases_db = data_root / "normalized" / "aliases.duckdb"
    raw_store = FileRawStore(shadow_root / "raw")
    event_store = DuckDbEventStore(shadow_root / "events.duckdb")
    normalizer = LdaFilingNormalizer()

    pages_fetched = 0
    filings_seen = 0
    new_filings_seen = 0
    unmapped = 0
    nonmodeled = 0
    enriched: list[Event] = []
    max_cursor = previous_cursor
    page = 1
    while True:
        raw = lda_client.fetch_filings_page(
            posted_after=posted_after,
            posted_before=posted_before,
            page=page,
            page_size=25,
        )
        raw_store.put(raw)
        pages_fetched += 1
        payload = _json_payload(raw.content)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("LDA shadow page response has no results list")
        filings_seen += len(results)

        company_map: dict[int, str] = {}
        new_filing_ids: set[str] = set()
        for filing in results:
            if not isinstance(filing, dict):
                continue
            cursor = _filing_cursor(filing)
            if cursor is None or cursor.posted_at > cutoff:
                continue
            if previous_cursor is not None and cursor <= previous_cursor:
                continue
            if previous_cursor is None and cursor.posted_at < cutoff - timedelta(days=initial_lookback_days):
                continue
            new_filings_seen += 1
            new_filing_ids.add(cursor.filing_uuid)
            if max_cursor is None or cursor > max_cursor:
                max_cursor = cursor

            client = filing.get("client") or {}
            if not isinstance(client, dict):
                unmapped += 1
                continue
            client_id = _int_or_none(client.get("id"))
            client_name = str(client.get("name") or "").strip()
            if client_id is None or not client_name:
                unmapped += 1
                continue
            resolved = _resolve_modeled_lda_company(
                aliases_db=aliases_db,
                client_id=client_id,
                client_name=client_name,
                modeled_ids=modeled_ids,
                name_index=name_index,
            )
            if resolved is None:
                unmapped += 1
                continue
            if resolved not in modeled_ids:
                nonmodeled += 1
                continue
            previous = company_map.get(client_id)
            if previous is not None and previous != resolved:
                raise ValueError(f"LDA client {client_id} resolved to multiple companies")
            company_map[client_id] = resolved

        for event in normalizer.to_events(raw, company_ids_by_client_id=company_map):
            if event.source_record_id not in new_filing_ids or event.company_id is None:
                continue
            semantic = extractor.extract(
                normalizer.semantic_text(event),
                context="US federal lobbying disclosure",
            )
            enriched.append(event.model_copy(update={"semantic": semantic}))

        next_url = payload.get("next")
        if not next_url:
            break
        if max_pages is not None and page >= max_pages:
            raise RuntimeError(
                "LDA shadow max_pages reached before pagination completed; intake cursor was not advanced"
            )
        page += 1

    # Persist normalized SHADOW events before advancing the intake cursor. A crash
    # before the atomic state write simply replays immutable IDs on the next poll.
    event_store.put_many(enriched)
    pending_added = intake.commit_poll(cursor=max_cursor, events=enriched)
    final_state = intake.load()
    return LdaShadowPollResult(
        pages_fetched=pages_fetched,
        filings_seen=filings_seen,
        new_filings_seen=new_filings_seen,
        mapped_event_count=len(enriched),
        semantic_enriched_event_count=sum(item.semantic is not None for item in enriched),
        unmapped_filing_count=unmapped,
        nonmodeled_filing_count=nonmodeled,
        pending_events_added=pending_added,
        pending_event_count=len(final_state.pending),
        cursor=final_state.cursor,
    )


def select_lda_shadow_batch(
    intake: FileLdaShadowIntake,
    *,
    resolver: XnysExecutionSessionResolver,
    as_of: datetime,
) -> LdaShadowBatchSelection:
    cutoff = as_utc(as_of)
    target = resolver.cycle_execution_date(cutoff)
    selected: list[str] = []
    stale: list[str] = []
    future: list[str] = []
    for item in intake.pending(as_of=cutoff):
        intended = resolver.execution_date(item.public_time)
        if intended < target:
            stale.append(item.event_id)
        elif intended > target:
            future.append(item.event_id)
        else:
            selected.append(item.event_id)
    return LdaShadowBatchSelection(
        target_execution_date=target,
        selected_event_ids=tuple(selected),
        stale_event_ids=tuple(stale),
        future_event_ids=tuple(future),
    )


def run_current_lda_shadow(
    *,
    data_root: str | Path = "data",
    experiment_dir: str | Path = "data/experiments/lightgbm_holdout_250_v2",
    runtime_dir: str | Path = "data/runtime",
    as_of: datetime | None = None,
    initial_lookback_days: int = 2,
    max_pages: int | None = None,
    qwen_base_url: str | None = None,
    qwen_model: str | None = None,
    lda_api_token: str | None = None,
) -> dict:
    """Collect and score live LDA disclosures with zero PAPER authority."""

    data_root = Path(data_root)
    experiment_dir = Path(experiment_dir)
    runtime_root = Path(runtime_dir)
    cutoff = as_utc(as_of or datetime.now(timezone.utc))
    shadow_root = runtime_root / "lda_shadow"
    resolved_qwen_url = (
        qwen_base_url or os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
    ).strip()
    resolved_qwen_model = (
        qwen_model or os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-4B")
    ).strip()
    token = (lda_api_token or os.environ.get("LDA_API_TOKEN", "")).strip() or None

    resolver = XnysExecutionSessionResolver()
    config = _load_runtime_config(runtime_root)
    market_store = DuckDbMarketStore(Path(str(config["market_db"])))
    market_store.enable_read_cache(max_series=32)
    intake = FileLdaShadowIntake(shadow_root / "intake.json")

    try:
        with LdaClient(api_key=token) as lda_client, QwenSemanticExtractor(
            cache=FileSemanticCache(data_root / "cache" / "semantic"),
            base_url=resolved_qwen_url,
            model=resolved_qwen_model,
            extractor_version="semantic-v1",
        ) as extractor:
            poll = poll_current_lda_shadow(
                data_root=data_root,
                experiment_dir=experiment_dir,
                runtime_dir=runtime_root,
                lda_client=lda_client,
                extractor=extractor,
                as_of=cutoff,
                initial_lookback_days=initial_lookback_days,
                max_pages=max_pages,
            )
    except httpx.RequestError as exc:
        raise RuntimeError(
            "LDA shadow semantic enrichment requires the configured local Qwen OpenAI-compatible endpoint"
        ) from exc

    selection = select_lda_shadow_batch(intake, resolver=resolver, as_of=cutoff)
    stale_removed = intake.dispose_stale(selection.stale_event_ids)
    base_payload = {
        "as_of": cutoff.isoformat(),
        "authority": "shadow_only_no_paper",
        "evidence_source": "lda_shadow",
        "qwen_model": resolved_qwen_model,
        "poll": _jsonable(asdict(poll)),
        "selection": {
            "target_execution_date": selection.target_execution_date.isoformat(),
            "selected_event_count": len(selection.selected_event_ids),
            "selected_event_ids": list(selection.selected_event_ids),
            "stale_event_count": len(selection.stale_event_ids),
            "stale_event_ids": list(selection.stale_event_ids),
            "stale_events_disposed": stale_removed,
            "future_event_count": len(selection.future_event_ids),
            "future_event_ids": list(selection.future_event_ids),
        },
    }
    if not selection.selected_event_ids:
        scorecard = refresh_forward_outcome_scorecard(
            data_root=data_root,
            runtime_dir=runtime_root,
            as_of=cutoff,
        )
        return {
            **base_payload,
            "status": "no_actionable_lda_shadow_batch",
            "scored_event_count": 0,
            "remaining_pending_count": len(intake.pending()),
            "forward_outcomes": scorecard,
        }

    shadow_event_store = DuckDbEventStore(shadow_root / "events.duckdb")
    selected_events = _events_by_ids(
        shadow_event_store,
        selection.selected_event_ids,
    )
    selected_company_ids = tuple(sorted({item.company_id for item in selected_events if item.company_id}))
    credentials = _load_tiingo_token(data_root)
    completed_session = resolver.last_completed_session(cutoff)
    shadow_raw_store = FileRawStore(shadow_root / "raw")
    with TiingoClient(credentials) as tiingo_client:
        market_sync = _sync_known_companies(
            selected_company_ids,
            target_execution_date=selection.target_execution_date,
            completed_session=completed_session,
            benchmark_security_id=str(config["benchmark_security_id"]),
            market_store=market_store,
            tiingo_client=tiingo_client,
            raw_store=shadow_raw_store,
        )
    market_store.clear_read_cache()
    ready_company_ids = tuple(market_sync["ready_company_ids"])
    ready_set = set(ready_company_ids)
    ready_events = tuple(
        event for event in selected_events if event.company_id in ready_set
    )
    unready_event_ids = tuple(
        event.event_id for event in selected_events if event.company_id not in ready_set
    )
    if not ready_events:
        return {
            **base_payload,
            "status": "no_market_ready_lda_shadow_batch",
            "market_sync": market_sync,
            "scored_event_count": 0,
            "unready_event_ids": list(unready_event_ids),
            "remaining_pending_count": len(intake.pending()),
        }

    current_batch_id = batch_id(
        selection.target_execution_date,
        tuple(sorted(event.event_id for event in ready_events)),
    )
    diagnostic_root = runtime_root / "decision_diagnostics"
    diagnostic_path = diagnostic_root / f"{current_batch_id}.json"
    if diagnostic_path.exists():
        acknowledged = intake.acknowledge(event.event_id for event in ready_events)
        scorecard = refresh_forward_outcome_scorecard(
            data_root=data_root,
            runtime_dir=runtime_root,
            as_of=cutoff,
        )
        return {
            **base_payload,
            "status": "completed_from_existing_lda_shadow_diagnostic",
            "batch_id": current_batch_id,
            "diagnostic_path": str(diagnostic_path),
            "market_sync": market_sync,
            "scored_event_count": len(ready_events),
            "unready_event_ids": list(unready_event_ids),
            "acknowledged_scored_event_count": acknowledged,
            "remaining_pending_count": len(intake.pending()),
            "forward_outcomes": scorecard,
        }

    authoritative_events = DuckDbEventStore(
        data_root / "normalized" / "events.duckdb"
    ).all_events(company_ids=list(ready_company_ids))
    shadow_history = shadow_event_store.all_events(company_ids=list(ready_company_ids))
    all_events = _merge_event_history(authoritative_events, shadow_history)
    snapshot_builder = CandidateSnapshotBuilder(
        market_store,
        benchmark_security_id=str(config["benchmark_security_id"]),
    )
    assembly = PitCandidateAssembler(snapshot_builder).assemble(
        ready_events,
        all_events=all_events,
        as_of=cutoff,
        execution_date=selection.target_execution_date,
    )
    if assembly.candidate_count != assembly.affected_company_count:
        raise RuntimeError(
            "LDA shadow candidate assembly is incomplete; selected events remain pending"
        )

    loaded = load_persisted_shadow_registry(runtime_dir=runtime_root)
    restored_strategy_ids = FileRuntimeStrategyStateStore(
        runtime_root / "strategy_state"
    ).restore_registry(loaded.registry)
    diagnostics = diagnose_registry(loaded.registry, assembly.candidates)
    written_path = FileStrategyDecisionDiagnosticStore(diagnostic_root).write(
        batch_id=current_batch_id,
        as_of=cutoff,
        target_execution_date=selection.target_execution_date,
        diagnostics=diagnostics,
        evidence_source="lda_shadow",
    )
    acknowledged = intake.acknowledge(event.event_id for event in ready_events)
    scorecard = refresh_forward_outcome_scorecard(
        data_root=data_root,
        runtime_dir=runtime_root,
        as_of=cutoff,
    )
    return {
        **base_payload,
        "status": "completed",
        "batch_id": current_batch_id,
        "diagnostic_path": str(written_path),
        "market_sync": market_sync,
        "candidate_assembly": {
            "trigger_event_count": assembly.trigger_event_count,
            "affected_company_count": assembly.affected_company_count,
            "candidate_count": assembly.candidate_count,
            "candidate_ids": [item.candidate_id for item in assembly.candidates],
        },
        "restored_strategy_ids": list(restored_strategy_ids),
        "strategies": [
            {
                "strategy_id": item.strategy_id,
                "candidate_count": item.candidate_count,
                "emitted_opportunity_count": item.emitted_opportunity_count,
            }
            for item in diagnostics
        ],
        "scored_event_count": len(ready_events),
        "unready_event_ids": list(unready_event_ids),
        "acknowledged_scored_event_count": acknowledged,
        "remaining_pending_count": len(intake.pending()),
        "forward_outcomes": scorecard,
    }


def _sync_known_companies(
    company_ids: tuple[str, ...],
    *,
    target_execution_date: date,
    completed_session: date,
    benchmark_security_id: str,
    market_store: DuckDbMarketStore,
    tiingo_client: TiingoClient,
    raw_store: FileRawStore,
) -> dict:
    ready: list[str] = []
    failures: list[dict] = []
    downloaded_series = 0
    bars_added = 0
    for company_id in company_ids:
        security_id = market_store.security_for_company(company_id, target_execution_date)
        if security_id is None:
            failures.append({"company_id": company_id, "reason": "no_verified_current_security_mapping"})
            continue
        latest = market_store.bars_before(
            security_id,
            completed_session + timedelta(days=1),
            1,
        )
        if not latest:
            failures.append({"company_id": company_id, "reason": "no_stored_market_history"})
            continue
        ticker = latest[-1].ticker
        try:
            downloaded, added = _sync_known_series(
                security_id,
                ticker,
                completed_session=completed_session,
                market_store=market_store,
                tiingo_client=tiingo_client,
                raw_store=raw_store,
            )
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            failures.append(
                {
                    "company_id": company_id,
                    "security_id": security_id,
                    "ticker": ticker,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        downloaded_series += int(downloaded)
        bars_added += added
        ready.append(company_id)

    benchmark_downloaded, benchmark_added = _sync_known_series(
        benchmark_security_id,
        "SPY",
        completed_session=completed_session,
        market_store=market_store,
        tiingo_client=tiingo_client,
        raw_store=raw_store,
    )
    return {
        "completed_session": completed_session.isoformat(),
        "target_execution_date": target_execution_date.isoformat(),
        "company_count": len(company_ids),
        "ready_company_count": len(ready),
        "ready_company_ids": ready,
        "unready_company_count": len(company_ids) - len(ready),
        "downloaded_price_series": downloaded_series,
        "bars_added": bars_added,
        "benchmark_downloaded": benchmark_downloaded,
        "benchmark_bars_added": benchmark_added,
        "failures": failures,
    }


def _sync_known_series(
    security_id: str,
    ticker: str,
    *,
    completed_session: date,
    market_store: DuckDbMarketStore,
    tiingo_client: TiingoClient,
    raw_store: FileRawStore,
) -> tuple[bool, int]:
    bounds = market_store.date_bounds(security_id, ticker)
    if bounds is None:
        raise RuntimeError(f"verified security has no stored series: {security_id}/{ticker}")
    if bounds[1] >= completed_session:
        return False, 0
    start = bounds[1] + timedelta(days=1)
    raw = tiingo_client.fetch_prices(ticker, start, completed_session)
    raw_store.put(raw)
    bars = TiingoNormalizer().parse_prices(raw, security_id=security_id, ticker=ticker)
    before = market_store.count_bars(security_id, ticker, start, completed_session)
    market_store.put_many(bars)
    after = market_store.count_bars(security_id, ticker, start, completed_session)
    refreshed = market_store.date_bounds(security_id, ticker)
    if refreshed is None or refreshed[1] < completed_session:
        raise RuntimeError(f"Tiingo series still lags completed session: {security_id}/{ticker}")
    return True, max(0, after - before)


def _events_by_ids(store: DuckDbEventStore, event_ids: tuple[str, ...]) -> tuple[Event, ...]:
    wanted = set(event_ids)
    events = tuple(event for event in store.all_events() if event.event_id in wanted)
    by_id = {event.event_id: event for event in events}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise RuntimeError(f"LDA shadow normalized events are missing: {missing[:5]}")
    return tuple(sorted(events, key=lambda item: (item.public_time, item.event_id)))


def _merge_event_history(*groups: Iterable[Event]) -> tuple[Event, ...]:
    by_id: dict[str, Event] = {}
    for group in groups:
        for event in group:
            previous = by_id.get(event.event_id)
            if previous is None or (previous.semantic is None and event.semantic is not None):
                by_id[event.event_id] = event
    return tuple(sorted(by_id.values(), key=lambda item: (item.public_time, item.event_id)))


def _modeled_company_ids(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"missing training rows: {path}") from exc
    result: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            result.add(str(json.loads(line)["company_id"]))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid training row {line_number}: {path}") from exc
    if not result:
        raise ValueError("training rows contain no modeled company IDs")
    return result


def _modeled_company_name_index(path: Path, modeled_ids: set[str]) -> dict[str, tuple[str, ...]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"missing SEC company manifest: {path}") from exc
    values: dict[str, set[str]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            company_id = str(row["company_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid SEC company manifest row {line_number}: {path}") from exc
        if company_id not in modeled_ids:
            continue
        for raw_name in row.get("issuer_names", ()):
            name = normalize_company_name(str(raw_name))
            if name:
                values.setdefault(name, set()).add(company_id)
    return {name: tuple(sorted(ids)) for name, ids in values.items()}


def _resolve_modeled_lda_company(
    *,
    aliases_db: Path,
    client_id: int,
    client_name: str,
    modeled_ids: set[str],
    name_index: dict[str, tuple[str, ...]],
) -> str | None:
    alias = _read_lda_alias(aliases_db, str(client_id))
    if alias is not None:
        return alias if alias in modeled_ids else None
    matches = name_index.get(normalize_company_name(client_name), ())
    return matches[0] if len(matches) == 1 else None


def _read_lda_alias(path: Path, external_id: str) -> str | None:
    if not path.exists():
        return None
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for LDA alias lookup") from exc
    with duckdb.connect(str(path), read_only=True) as connection:
        table = connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'external_entity_aliases'
            """
        ).fetchone()
        if table is None or int(table[0]) == 0:
            return None
        row = connection.execute(
            """
            SELECT company_id FROM external_entity_aliases
            WHERE source = ? AND external_id = ?
            """,
            [Source.LDA.value, external_id],
        ).fetchone()
    return str(row[0]) if row is not None else None


def _filing_cursor(filing: dict) -> LdaFilingCursor | None:
    filing_uuid = str(filing.get("filing_uuid") or "").strip()
    posted = str(filing.get("dt_posted") or "").strip()
    if not filing_uuid or not posted:
        return None
    normalized = posted[:-1] + "+00:00" if posted.endswith("Z") else posted
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("LDA dt_posted must include a timezone")
    return LdaFilingCursor(as_utc(parsed), filing_uuid)


def _int_or_none(value) -> int | None:
    if value is None or not str(value).strip():
        return None
    return int(value)


def _json_payload(content: bytes | str) -> dict:
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("expected LDA JSON object")
    return payload


def _load_tiingo_token(data_root: Path) -> str:
    from stock_trading.local_secrets import load_tiingo_credentials

    return load_tiingo_credentials(data_root).token


def _pending_sort_key(item: LdaShadowPending):
    return item.public_time, item.event_id


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


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poll live LDA lobbying disclosures into isolated SHADOW storage, score "
            "them read-only with the current strategy registry, and publish forward evidence."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("data/experiments/lightgbm_holdout_250_v2"),
    )
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--initial-lookback-days", type=int, default=2)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--as-of")
    parser.add_argument(
        "--qwen-base-url",
        default=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-4B"),
    )
    args = parser.parse_args()
    result = run_current_lda_shadow(
        data_root=args.data_root,
        experiment_dir=args.experiment_dir,
        runtime_dir=args.runtime_dir,
        as_of=_parse_as_of(args.as_of),
        initial_lookback_days=args.initial_lookback_days,
        max_pages=args.max_pages,
        qwen_base_url=args.qwen_base_url,
        qwen_model=args.qwen_model,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
