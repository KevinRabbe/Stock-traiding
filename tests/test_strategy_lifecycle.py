from __future__ import annotations

import pytest

from stock_trading.engine import (
    ProfitabilityGate,
    StrategyLifecycleController,
    StrategyLifecyclePolicy,
    StrategyRecord,
    StrategyRegistry,
    StrategyScorecard,
    StrategyStage,
)


class _Strategy:
    def __init__(self, strategy_id):
        self._strategy_id = strategy_id

    @property
    def strategy_id(self):
        return self._strategy_id

    def evaluate(self, candidates, portfolio):
        del candidates, portfolio
        return ()


def _scorecard(*, return_=0.05, pf=1.5, trades=100, dd=0.02, year_rate=0.6):
    return StrategyScorecard(
        compounded_return=return_,
        profit_factor=pf,
        worst_realized_drawdown=dd,
        total_trades=trades,
        profitable_year_rate=year_rate,
    )


def _record(strategy_id, stage, scorecard=None, *, artifact_ref="artifacts/manifest.json"):
    return StrategyRecord(
        strategy_id=strategy_id,
        stage=stage,
        artifact_ref=artifact_ref,
        scorecard=scorecard,
    )


def test_lifecycle_requires_staged_forward_promotion_and_never_auto_switches_champion() -> None:
    registry = StrategyRegistry()
    incumbent = _Strategy("incumbent")
    challenger = _Strategy("challenger")
    registry.register(
        incumbent,
        _record("incumbent", StrategyStage.PAPER, _scorecard()),
    )
    registry.register(
        challenger,
        _record("challenger", StrategyStage.DEVELOPMENT, _scorecard()),
    )
    registry.set_champion("incumbent")
    controller = StrategyLifecycleController(registry)

    with pytest.raises(ValueError, match="invalid strategy lifecycle transition"):
        controller.transition("challenger", StrategyStage.PAPER)

    controller.transition("challenger", StrategyStage.SHADOW)
    controller.transition("challenger", StrategyStage.PAPER)
    controller.transition("challenger", StrategyStage.LIVE)

    assert registry.record("challenger").stage is StrategyStage.LIVE
    assert registry.champion_id == "incumbent"

    controller.promote_champion("challenger", required_stage=StrategyStage.LIVE)
    assert registry.champion_id == "challenger"


def test_lifecycle_profitability_gate_blocks_forward_transition() -> None:
    registry = StrategyRegistry()
    strategy = _Strategy("weak")
    registry.register(
        strategy,
        _record("weak", StrategyStage.DEVELOPMENT, _scorecard(return_=-0.01, pf=0.8)),
    )
    controller = StrategyLifecycleController(
        registry,
        StrategyLifecyclePolicy(
            shadow_gate=ProfitabilityGate(
                min_compounded_return=0.0,
                min_profit_factor=1.0,
                min_trades=50,
            )
        ),
    )

    with pytest.raises(ValueError, match="profitability gate"):
        controller.transition("weak", StrategyStage.SHADOW)
    assert registry.record("weak").stage is StrategyStage.DEVELOPMENT


def test_paper_promotion_requires_immutable_artifact_reference() -> None:
    registry = StrategyRegistry()
    strategy = _Strategy("no-artifact")
    registry.register(
        strategy,
        _record(
            "no-artifact",
            StrategyStage.SHADOW,
            _scorecard(),
            artifact_ref=None,
        ),
    )
    controller = StrategyLifecycleController(registry)

    with pytest.raises(ValueError, match="artifact_ref"):
        controller.transition("no-artifact", StrategyStage.PAPER)


def test_live_to_paper_safety_downgrade_does_not_require_gate() -> None:
    registry = StrategyRegistry()
    strategy = _Strategy("live")
    registry.register(
        strategy,
        _record("live", StrategyStage.LIVE, _scorecard(return_=-0.50, pf=0.1)),
    )
    controller = StrategyLifecycleController(
        registry,
        StrategyLifecyclePolicy(
            paper_gate=ProfitabilityGate(
                min_compounded_return=0.50,
                min_profit_factor=5.0,
                min_trades=10_000,
            )
        ),
    )

    updated = controller.transition("live", StrategyStage.PAPER)

    assert updated.stage is StrategyStage.PAPER


def test_champion_promotion_requires_explicit_requested_stage() -> None:
    registry = StrategyRegistry()
    strategy = _Strategy("paper")
    registry.register(
        strategy,
        _record("paper", StrategyStage.PAPER, _scorecard()),
    )
    controller = StrategyLifecycleController(registry)

    with pytest.raises(ValueError, match="must be live"):
        controller.promote_champion("paper", required_stage=StrategyStage.LIVE)

    controller.promote_champion("paper", required_stage=StrategyStage.PAPER)
    assert registry.champion_id == "paper"
