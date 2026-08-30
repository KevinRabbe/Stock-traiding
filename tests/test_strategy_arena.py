from datetime import UTC, datetime

import pytest

from stock_trading.research.strategy_arena import (
    ArenaPolicy,
    MarketStateSnapshot,
    StrategyArena,
    StrategyLifecycle,
    StrategyObservation,
)


def _market(*, confidence: float = 0.80) -> MarketStateSnapshot:
    return MarketStateSnapshot(
        as_of=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
        regime_id="trend-high-liquidity",
        confidence=confidence,
        dimensions={
            "trend_strength": 0.75,
            "volatility": 0.55,
            "liquidity": 0.90,
        },
    )


def _observation(
    strategy_id: str = "trend-01",
    *,
    family: str = "trend",
    trade_count: int = 120,
    expectancy: float = 0.004,
    recent_expectancy: float = 0.003,
    profit_factor: float = 1.45,
    max_drawdown: float = 0.02,
    regime_similarity: float = 0.85,
    stability: float = 0.80,
    invalidated: bool = False,
) -> StrategyObservation:
    return StrategyObservation(
        strategy_id=strategy_id,
        family=family,
        trade_count=trade_count,
        expectancy=expectancy,
        recent_expectancy=recent_expectancy,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        regime_similarity=regime_similarity,
        stability=stability,
        average_alpha=0.001,
        invalidated=invalidated,
    )


def _lifecycle(decision, strategy_id: str) -> StrategyLifecycle:
    return next(
        item.lifecycle
        for item in decision.strategies
        if item.strategy_id == strategy_id
    )


def test_strong_strategy_must_pass_shadow_before_receiving_capital() -> None:
    arena = StrategyArena()
    observation = _observation()

    first = arena.evaluate(_market(), (observation,))
    assert _lifecycle(first, "trend-01") is StrategyLifecycle.SHADOW
    assert first.allocations == ()

    second = arena.evaluate(
        _market(),
        (observation,),
        previous_states={"trend-01": StrategyLifecycle.SHADOW},
    )
    assert _lifecycle(second, "trend-01") is StrategyLifecycle.ACTIVE
    assert len(second.allocations) == 1
    assert second.allocations[0].strategy_id == "trend-01"
    assert second.allocations[0].weight == pytest.approx(0.25)
    assert second.cash_reserve_weight == pytest.approx(0.75)


def test_active_strategy_goes_dormant_when_current_regime_no_longer_matches() -> None:
    arena = StrategyArena()
    mismatch = _observation(regime_similarity=0.20)

    decision = arena.evaluate(
        _market(),
        (mismatch,),
        previous_states={"trend-01": StrategyLifecycle.ACTIVE},
    )

    assert _lifecycle(decision, "trend-01") is StrategyLifecycle.DORMANT
    assert decision.allocations == ()
    state = decision.strategies[0]
    assert "regime_fit" in state.reason


def test_dormant_strategy_can_reactivate_when_regime_returns() -> None:
    arena = StrategyArena()

    decision = arena.evaluate(
        _market(),
        (_observation(),),
        previous_states={"trend-01": StrategyLifecycle.DORMANT},
    )

    assert _lifecycle(decision, "trend-01") is StrategyLifecycle.ACTIVE
    assert decision.allocations[0].strategy_id == "trend-01"


def test_bad_performance_dormancy_does_not_permanently_delete_strategy() -> None:
    arena = StrategyArena()
    weak = _observation(
        expectancy=-0.001,
        recent_expectancy=-0.003,
        profit_factor=0.80,
    )

    decision = arena.evaluate(
        _market(),
        (weak,),
        previous_states={"trend-01": StrategyLifecycle.ACTIVE},
    )

    assert _lifecycle(decision, "trend-01") is StrategyLifecycle.DORMANT
    assert decision.strategies[0].lifecycle is not StrategyLifecycle.RETIRED


def test_structural_invalidation_retires_strategy() -> None:
    arena = StrategyArena()

    decision = arena.evaluate(
        _market(),
        (_observation(invalidated=True),),
        previous_states={"trend-01": StrategyLifecycle.ACTIVE},
    )

    assert _lifecycle(decision, "trend-01") is StrategyLifecycle.RETIRED
    assert decision.allocations == ()


def test_low_market_state_confidence_prevents_active_promotion() -> None:
    arena = StrategyArena()

    decision = arena.evaluate(
        _market(confidence=0.30),
        (_observation(),),
        previous_states={"trend-01": StrategyLifecycle.SHADOW},
    )

    assert _lifecycle(decision, "trend-01") is StrategyLifecycle.SHADOW
    assert decision.allocations == ()
    assert "market_state_confidence" in decision.strategies[0].reason


def test_allocator_caps_each_strategy_and_preserves_total_risk_reserve() -> None:
    policy = ArenaPolicy(
        max_active_strategies=4,
        max_strategy_weight=0.25,
        max_total_weight=0.80,
    )
    arena = StrategyArena(policy)
    observations = tuple(
        _observation(
            f"strategy-{index}",
            family="mixed",
            expectancy=0.002 + index * 0.0005,
            recent_expectancy=0.001 + index * 0.0004,
            profit_factor=1.20 + index * 0.05,
            max_drawdown=0.04 - index * 0.004,
            regime_similarity=0.65 + index * 0.05,
            stability=0.60 + index * 0.05,
        )
        for index in range(6)
    )
    previous = {
        item.strategy_id: StrategyLifecycle.SHADOW
        for item in observations
    }

    decision = arena.evaluate(
        _market(),
        observations,
        previous_states=previous,
    )

    assert len(decision.allocations) == 4
    assert sum(item.weight for item in decision.allocations) == pytest.approx(0.80)
    assert all(item.weight <= 0.25 + 1e-12 for item in decision.allocations)
    assert decision.cash_reserve_weight == pytest.approx(0.20)
    allocated_ids = {item.strategy_id for item in decision.allocations}
    assert "strategy-5" in allocated_ids
    assert "strategy-4" in allocated_ids


def test_arena_rejects_duplicate_strategy_identity() -> None:
    arena = StrategyArena()

    with pytest.raises(ValueError, match="duplicate strategy_id"):
        arena.evaluate(
            _market(),
            (_observation("same"), _observation("same")),
        )


def test_market_state_rejects_nonfinite_dimensions() -> None:
    with pytest.raises(ValueError, match="finite"):
        MarketStateSnapshot(
            as_of=datetime(2026, 8, 30, tzinfo=UTC),
            regime_id="bad",
            confidence=0.5,
            dimensions={"volatility": float("nan")},
        )
