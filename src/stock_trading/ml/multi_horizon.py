from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Iterable, Mapping

from stock_trading.market.execution_time import decision_market_date
from stock_trading.market.labels import build_standard_labels
from stock_trading.market.store import DuckDbMarketStore

from .dataset import TrainingRow


@dataclass(frozen=True, slots=True)
class HorizonTarget:
    horizon: int
    exit_date: date
    stock_return: float
    benchmark_return: float
    alpha: float
    downside: float
    mfe: float


def build_multi_horizon_targets(
    rows: Iterable[TrainingRow],
    market_store: DuckDbMarketStore,
    *,
    benchmark_security_id: str,
    horizons: tuple[int, ...] = (5, 20, 60),
    verify_existing_20d: bool = True,
) -> dict[str, dict[int, HorizonTarget]]:
    """Reconstruct forward labels for existing PIT rows from the local market DB.

    This deliberately does not alter model inputs. It only recovers outcomes that
    were already available upstream when the original dataset was built but were
    discarded because ``TrainingRow`` retained only the 20-session target.

    Rows missing any requested horizon are omitted entirely. That prevents future
    label maturity from influencing the adaptive horizon decision inside V5.
    """

    requested = tuple(sorted(set(int(horizon) for horizon in horizons)))
    if not requested or any(horizon <= 0 for horizon in requested):
        raise ValueError("horizons must contain positive integers")
    if verify_existing_20d and 20 not in requested:
        raise ValueError("20 must be requested when verify_existing_20d is enabled")

    max_horizon = max(requested)
    result: dict[str, dict[int, HorizonTarget]] = {}
    for row in rows:
        mapping_day = decision_market_date(row.decision_time)
        security_id = market_store.security_for_company(row.company_id, mapping_day)
        if security_id is None:
            continue

        stock_future = market_store.bars_from(
            security_id,
            row.execution_date,
            max_horizon,
        )
        benchmark_future = market_store.bars_from(
            benchmark_security_id,
            row.execution_date,
            max_horizon,
        )
        labels = build_standard_labels(
            stock_future,
            benchmark_future,
            horizons=requested,
        )
        by_horizon = {label.horizon: label for label in labels}
        if any(horizon not in by_horizon for horizon in requested):
            continue

        if verify_existing_20d:
            _verify_twenty_day_identity(row, by_horizon[20])

        result[row.event_id] = {
            horizon: HorizonTarget(
                horizon=horizon,
                exit_date=by_horizon[horizon].end_date,
                stock_return=by_horizon[horizon].stock_return,
                benchmark_return=by_horizon[horizon].benchmark_return,
                alpha=by_horizon[horizon].alpha,
                downside=max(0.0, -by_horizon[horizon].max_adverse_excursion),
                mfe=max(0.0, by_horizon[horizon].max_favorable_excursion),
            )
            for horizon in requested
        }
    return result


def multi_horizon_maturity_dates(
    rows: Iterable[TrainingRow],
    targets: Mapping[str, Mapping[int, HorizonTarget]],
    *,
    horizons: tuple[int, ...],
) -> dict[str, date]:
    """Return the latest realized-label date required by each strategy row.

    A multi-horizon model is point-in-time safe only after every target it trains
    or calibrates against has matured. The latest requested horizon exit is thus
    the row's effective maturity fence for walk-forward train/validation splits.
    """

    requested = tuple(sorted(set(int(horizon) for horizon in horizons)))
    if not requested or any(horizon <= 0 for horizon in requested):
        raise ValueError("horizons must contain positive integers")

    result: dict[str, date] = {}
    for row in rows:
        by_horizon = targets.get(row.event_id)
        if by_horizon is None:
            raise ValueError(f"missing multi-horizon targets for {row.event_id}")
        missing = [horizon for horizon in requested if horizon not in by_horizon]
        if missing:
            raise ValueError(f"missing horizons {missing} for {row.event_id}")
        result[row.event_id] = max(
            by_horizon[horizon].exit_date for horizon in requested
        )
    return result


def row_for_horizon(
    row: TrainingRow,
    target: HorizonTarget,
    *,
    positive_alpha_threshold: float = 0.02,
) -> TrainingRow:
    """Project a generic horizon target through the legacy 20d row interface."""

    return replace(
        row,
        exit_date_20d=target.exit_date,
        stock_return_20d=target.stock_return,
        benchmark_return_20d=target.benchmark_return,
        alpha_20d=target.alpha,
        downside_20d=target.downside,
        mfe_20d=target.mfe,
        positive_alpha_20d=int(target.alpha >= positive_alpha_threshold),
    )


def _verify_twenty_day_identity(row: TrainingRow, label) -> None:
    if label.start_date != row.execution_date or label.end_date != row.exit_date_20d:
        raise ValueError(
            f"reconstructed 20d dates diverge for {row.event_id}: "
            f"{label.start_date}/{label.end_date} != "
            f"{row.execution_date}/{row.exit_date_20d}"
        )
    checks = (
        ("stock_return", label.stock_return, row.stock_return_20d),
        ("benchmark_return", label.benchmark_return, row.benchmark_return_20d),
        ("alpha", label.alpha, row.alpha_20d),
        ("downside", max(0.0, -label.max_adverse_excursion), row.downside_20d),
        ("mfe", max(0.0, label.max_favorable_excursion), row.mfe_20d),
    )
    for name, reconstructed, existing in checks:
        if abs(float(reconstructed) - float(existing)) > 1e-10:
            raise ValueError(
                f"reconstructed 20d {name} diverges for {row.event_id}: "
                f"{reconstructed} != {existing}"
            )
