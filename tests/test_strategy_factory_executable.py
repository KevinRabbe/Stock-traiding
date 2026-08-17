from __future__ import annotations

from types import SimpleNamespace

from stock_trading.experiments.lightgbm_strategy_factory_executable import (
    _ExecutableWorkerContext,
    _filter_executable_rows,
)
from stock_trading.research.strategy_factory import StrategyVariantSpec


def test_execution_filters_quality_before_pit_liquidity() -> None:
    rows = (
        SimpleNamespace(
            event_id="executable",
            features={"market.avg_dollar_volume_20d": 100_000.0},
        ),
        SimpleNamespace(
            event_id="illiquid",
            features={"market.avg_dollar_volume_20d": 1_000.0},
        ),
        SimpleNamespace(
            event_id="bad-quality",
            features={"market.avg_dollar_volume_20d": 100_000.0},
        ),
    )
    context = _ExecutableWorkerContext(
        rows=rows,
        targets={},
        security_ids={},
        entry_liquidity={},
        invalid_target_keys=frozenset({("bad-quality", 20)}),
        starting_capital=10_000.0,
        allocation_pct=0.02,
        max_open_positions=15,
        round_trip_cost_bps=20.0,
        min_train_rows=100,
        max_trailing_adv_participation_pct=0.01,
        max_entry_day_participation_pct=0.01,
    )
    spec = StrategyVariantSpec(
        variant_id="test",
        feature_profile="event_history",
        training_window_years=None,
        tree_profile="baseline",
        horizons=(20,),
        alpha_rank_weight=0.25,
        seed=42,
    )

    executable, quality_removed, liquidity_removed = _filter_executable_rows(
        rows,
        spec,
        context,
    )

    assert [row.event_id for row in executable] == ["executable"]
    assert quality_removed == 1
    assert liquidity_removed == 1


def test_quality_filter_is_horizon_specific() -> None:
    row = SimpleNamespace(
        event_id="candidate",
        features={"market.avg_dollar_volume_20d": 100_000.0},
    )
    context = _ExecutableWorkerContext(
        rows=(row,),
        targets={},
        security_ids={},
        entry_liquidity={},
        invalid_target_keys=frozenset({("candidate", 60)}),
        starting_capital=10_000.0,
        allocation_pct=0.02,
        max_open_positions=15,
        round_trip_cost_bps=20.0,
        min_train_rows=100,
        max_trailing_adv_participation_pct=0.01,
        max_entry_day_participation_pct=0.01,
    )
    short_spec = StrategyVariantSpec(
        variant_id="short",
        feature_profile="event_history",
        training_window_years=None,
        tree_profile="baseline",
        horizons=(5, 20),
        alpha_rank_weight=0.25,
        seed=42,
    )
    long_spec = StrategyVariantSpec(
        variant_id="long",
        feature_profile="event_history",
        training_window_years=None,
        tree_profile="baseline",
        horizons=(20, 60),
        alpha_rank_weight=0.25,
        seed=42,
    )

    short_rows, short_quality_removed, _ = _filter_executable_rows((row,), short_spec, context)
    long_rows, long_quality_removed, _ = _filter_executable_rows((row,), long_spec, context)

    assert short_rows == (row,)
    assert short_quality_removed == 0
    assert long_rows == ()
    assert long_quality_removed == 1
