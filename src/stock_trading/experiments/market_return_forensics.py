from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from stock_trading.market import DuckDbMarketStore
from stock_trading.market.execution_time import decision_market_date
from stock_trading.market.models import MarketBar
from stock_trading.ml.multi_horizon import build_multi_horizon_targets

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class ReturnForensic:
    event_id: str
    company_id: str
    security_id: str
    ticker: str
    horizon: int
    entry_date: str
    exit_date: str
    adjusted_return: float
    raw_return: float
    entry_adj_open: float
    exit_adj_close: float
    entry_raw_open: float
    exit_raw_close: float
    entry_adjustment_factor: float
    exit_adjustment_factor: float
    adjustment_factor_ratio: float
    non_unit_split_factors: tuple[dict, ...]
    largest_adjusted_close_jump: dict | None
    largest_raw_close_jump: dict | None

    def as_json(self) -> dict:
        return {
            "event_id": self.event_id,
            "company_id": self.company_id,
            "security_id": self.security_id,
            "ticker": self.ticker,
            "horizon": self.horizon,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "adjusted_return": self.adjusted_return,
            "raw_return": self.raw_return,
            "entry_adj_open": self.entry_adj_open,
            "exit_adj_close": self.exit_adj_close,
            "entry_raw_open": self.entry_raw_open,
            "exit_raw_close": self.exit_raw_close,
            "entry_adjustment_factor": self.entry_adjustment_factor,
            "exit_adjustment_factor": self.exit_adjustment_factor,
            "adjustment_factor_ratio": self.adjustment_factor_ratio,
            "non_unit_split_factors": list(self.non_unit_split_factors),
            "largest_adjusted_close_jump": self.largest_adjusted_close_jump,
            "largest_raw_close_jump": self.largest_raw_close_jump,
        }


def run_market_return_forensics(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    horizons: tuple[int, ...] = (5, 20, 60),
    top_n: int = 30,
    extreme_positive_return: float = 2.0,
    extreme_negative_return: float = -0.90,
    market_read_cache_series: int = 200,
) -> Path:
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("horizons must contain positive values")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    if extreme_positive_return <= 0:
        raise ValueError("extreme_positive_return must be > 0")
    if not -1.0 < extreme_negative_return < 0.0:
        raise ValueError("extreme_negative_return must be in (-1, 0)")

    root = Path(experiment_dir)
    rows = _load_training_rows(root / "training_rows.jsonl")
    store = DuckDbMarketStore(market_db)
    store.enable_read_cache(max_series=market_read_cache_series)
    targets = build_multi_horizon_targets(
        rows,
        store,
        benchmark_security_id=benchmark_security_id,
        horizons=horizons,
        verify_existing_20d=20 in horizons,
    )

    findings: list[ReturnForensic] = []
    missing_security = 0
    missing_endpoint_bar = 0
    for row in rows:
        row_targets = targets.get(row.event_id)
        if row_targets is None:
            continue
        security_id = store.security_for_company(
            row.company_id,
            decision_market_date(row.decision_time),
        )
        if security_id is None:
            missing_security += 1
            continue
        for horizon in horizons:
            target = row_targets[horizon]
            entry = store.bar_on(security_id, row.execution_date)
            exit_bar = store.bar_on(security_id, target.exit_date)
            if entry is None or exit_bar is None:
                missing_endpoint_bar += 1
                continue
            series = tuple(
                bar
                for bar in store.bars_from(security_id, row.execution_date, max(horizons) + 10)
                if bar.date <= target.exit_date
            )
            findings.append(
                audit_target(
                    event_id=row.event_id,
                    company_id=row.company_id,
                    security_id=security_id,
                    horizon=horizon,
                    expected_adjusted_return=target.stock_return,
                    entry=entry,
                    exit_bar=exit_bar,
                    series=series,
                )
            )

    positive = sorted(findings, key=lambda item: item.adjusted_return, reverse=True)
    negative = sorted(findings, key=lambda item: item.adjusted_return)
    extreme = [
        item
        for item in findings
        if item.adjusted_return >= extreme_positive_return
        or item.adjusted_return <= extreme_negative_return
    ]
    extreme.sort(key=lambda item: abs(item.adjusted_return), reverse=True)

    payload = _json_safe(
        {
            "schema_version": "market-return-forensics-v1",
            "experiment_dir": str(root),
            "market_db": str(market_db),
            "benchmark_security_id": benchmark_security_id,
            "horizons": list(horizons),
            "row_count": len(rows),
            "audited_target_count": len(findings),
            "missing_security_count": missing_security,
            "missing_endpoint_bar_count": missing_endpoint_bar,
            "thresholds": {
                "extreme_positive_return": extreme_positive_return,
                "extreme_negative_return": extreme_negative_return,
            },
            "extreme_count": len(extreme),
            "extremes": [item.as_json() for item in extreme],
            "top_positive": [item.as_json() for item in positive[:top_n]],
            "top_negative": [item.as_json() for item in negative[:top_n]],
            "market_cache_stats": store.read_cache_stats(),
        }
    )
    output_path = root / "market_return_forensics.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def audit_target(
    *,
    event_id: str,
    company_id: str,
    security_id: str,
    horizon: int,
    expected_adjusted_return: float,
    entry: MarketBar,
    exit_bar: MarketBar,
    series: Iterable[MarketBar],
) -> ReturnForensic:
    entry_adj_open = float(entry.adj_open)
    exit_adj_close = float(exit_bar.adj_close)
    entry_raw_open = float(entry.open)
    exit_raw_close = float(exit_bar.close)
    adjusted_return = exit_adj_close / entry_adj_open - 1.0
    if abs(adjusted_return - expected_adjusted_return) > 1e-10:
        raise ValueError(
            f"target/bar mismatch for {event_id} {horizon}d: "
            f"{expected_adjusted_return} != {adjusted_return}"
        )
    raw_return = exit_raw_close / entry_raw_open - 1.0
    entry_factor = entry_adj_open / entry_raw_open
    exit_factor = exit_adj_close / exit_raw_close

    ordered = tuple(sorted(series, key=lambda item: item.date))
    splits = tuple(
        {
            "date": item.date.isoformat(),
            "split_factor": float(item.split_factor),
            "raw_close": float(item.close),
            "adj_close": float(item.adj_close),
        }
        for item in ordered
        if abs(float(item.split_factor) - 1.0) > 1e-12
    )
    return ReturnForensic(
        event_id=event_id,
        company_id=company_id,
        security_id=security_id,
        ticker=entry.ticker,
        horizon=horizon,
        entry_date=entry.date.isoformat(),
        exit_date=exit_bar.date.isoformat(),
        adjusted_return=adjusted_return,
        raw_return=raw_return,
        entry_adj_open=entry_adj_open,
        exit_adj_close=exit_adj_close,
        entry_raw_open=entry_raw_open,
        exit_raw_close=exit_raw_close,
        entry_adjustment_factor=entry_factor,
        exit_adjustment_factor=exit_factor,
        adjustment_factor_ratio=exit_factor / entry_factor,
        non_unit_split_factors=splits,
        largest_adjusted_close_jump=_largest_close_jump(ordered, adjusted=True),
        largest_raw_close_jump=_largest_close_jump(ordered, adjusted=False),
    )


def _largest_close_jump(
    bars: tuple[MarketBar, ...],
    *,
    adjusted: bool,
) -> dict | None:
    if len(bars) < 2:
        return None
    best: dict | None = None
    best_abs = -1.0
    for previous, current in zip(bars, bars[1:], strict=False):
        previous_close = float(previous.adj_close if adjusted else previous.close)
        current_close = float(current.adj_close if adjusted else current.close)
        value = current_close / previous_close - 1.0
        if abs(value) > best_abs:
            best_abs = abs(value)
            best = {
                "from_date": previous.date.isoformat(),
                "to_date": current.date.isoformat(),
                "return": value,
                "from_close": previous_close,
                "to_close": current_close,
                "to_split_factor": float(current.split_factor),
            }
    return best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit extreme historical 5/20/60-session returns and expose raw vs "
            "adjusted price/corporate-action discontinuities without training models."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--benchmark-security-id", default="benchmark_spy")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--extreme-positive-return", type=float, default=2.0)
    parser.add_argument("--extreme-negative-return", type=float, default=-0.90)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_path = run_market_return_forensics(
        args.experiment_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        horizons=tuple(args.horizons),
        top_n=args.top_n,
        extreme_positive_return=args.extreme_positive_return,
        extreme_negative_return=args.extreme_negative_return,
        market_read_cache_series=args.market_read_cache_series,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    compact = {
        "schema_version": payload["schema_version"],
        "audited_target_count": payload["audited_target_count"],
        "extreme_count": payload["extreme_count"],
        "output_path": str(output_path),
        "extremes": payload["extremes"],
    }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
