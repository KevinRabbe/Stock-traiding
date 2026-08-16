from __future__ import annotations

import pickle
from datetime import date, datetime, timezone

from stock_trading.experiments import lightgbm_strategy_factory as factory
from stock_trading.ml.dataset import TrainingRow


def _row() -> TrainingRow:
    return TrainingRow(
        event_id="event-1",
        company_id="company-1",
        decision_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        execution_date=date(2025, 1, 3),
        exit_date_20d=date(2025, 2, 3),
        features={"market.return_20d": 0.05},
        stock_return_20d=0.04,
        benchmark_return_20d=0.01,
        alpha_20d=0.03,
        downside_20d=0.02,
        mfe_20d=0.08,
        positive_alpha_20d=1,
    )


def test_prepared_factory_payload_is_spawn_picklable() -> None:
    row = _row()
    prepared = factory._PreparedFactoryData(
        rows=(row,),
        targets={},
        security_ids={row.event_id: "security-1"},
        market_cache_stats={"cached_series": 1},
    )

    restored = pickle.loads(pickle.dumps(prepared))

    assert restored == prepared
    assert restored.security_ids[row.event_id] == "security-1"


def test_worker_initialization_uses_parent_prepared_market_identity() -> None:
    row = _row()
    prepared = factory._PreparedFactoryData(
        rows=(row,),
        targets={},
        security_ids={row.event_id: "security-1"},
        market_cache_stats={},
    )
    common = {
        "starting_capital": 10_000.0,
        "allocation_pct": 0.02,
        "max_open_positions": 15,
        "round_trip_cost_bps": 20.0,
        "min_train_rows": 100,
        # Deliberately no market_db key: compute workers must not need one.
    }

    factory._initialize_worker(common, prepared)
    snapshots = factory._feature_snapshots((row,), prepared.security_ids)

    assert factory._CONTEXT is not None
    assert factory._CONTEXT.security_ids == {row.event_id: "security-1"}
    assert snapshots[0].security_id == "security-1"
    assert snapshots[0].candidate_id == row.event_id
