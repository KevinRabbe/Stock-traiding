from __future__ import annotations

from datetime import date, datetime, timezone

from stock_trading.engine import (
    BasicOpportunityRiskPolicy,
    ExecutionReport,
    ExecutionStatus,
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    HoldPositions,
    Opportunity,
    PortfolioSnapshot,
    StrategyRecord,
    StrategyRegistry,
    StrategyStage,
    TradingEngine,
)
from stock_trading.engine.policies import PassThroughPortfolioRiskPolicy
from stock_trading.live import ShadowStrategyEvaluator, TradingService


_NOW = datetime(2025, 1, 2, 18, tzinfo=timezone.utc)


class _Source:
    def __init__(self):
        self.calls = 0
        self.batch = (
            FeatureSnapshot(
                candidate_id="candidate-a",
                event_id="event-a",
                company_id="company-a",
                security_id="security-a",
                decision_time=_NOW,
                execution_date=_NOW.date(),
                features={"signal": 1.0},
            ),
        )

    def candidates(self, as_of):
        assert as_of == _NOW
        self.calls += 1
        return self.batch


class _State:
    def __init__(self):
        self.calls = 0
        self.snapshot_value = PortfolioSnapshot(
            as_of=_NOW,
            equity=10_000.0,
            cash=10_000.0,
            gross_exposure_pct=0.0,
        )

    def snapshot(self, as_of):
        assert as_of == _NOW
        self.calls += 1
        return self.snapshot_value


class _Strategy:
    def __init__(self, strategy_id, score):
        self._strategy_id = strategy_id
        self.score = score
        self.seen = []

    @property
    def strategy_id(self):
        return self._strategy_id

    def evaluate(self, candidates, portfolio):
        self.seen.append((candidates, portfolio))
        item = candidates[0]
        return (
            Opportunity(
                strategy_id=self.strategy_id,
                candidate_id=item.candidate_id,
                event_id=item.event_id,
                company_id=item.company_id,
                security_id=item.security_id,
                execution_date=item.execution_date,
                score=self.score,
                expected_return=0.03,
                expected_alpha=0.02,
                expected_downside=0.01,
                probability_positive=0.8,
                horizon_sessions=20,
            ),
        )


class _Broker:
    def __init__(self):
        self.settle_calls = 0
        self.orders = ()

    def settle(self, as_of):
        assert as_of == _NOW
        self.settle_calls += 1
        return ()

    def execute(self, orders):
        self.orders = orders
        return tuple(
            ExecutionReport(
                order_id=order.order_id,
                accepted=True,
                executed_at=_NOW,
                fill_price=100.0,
                status=ExecutionStatus.FILLED,
            )
            for order in orders
        )


def test_trading_service_shadows_and_champion_share_one_prepared_pit_context() -> None:
    source = _Source()
    state = _State()
    broker = _Broker()
    champion = _Strategy("champion", 0.99)
    shadow = _Strategy("shadow", 0.98)
    registry = StrategyRegistry()
    registry.register(
        champion,
        StrategyRecord(strategy_id="champion", stage=StrategyStage.PAPER),
    )
    registry.register(
        shadow,
        StrategyRecord(strategy_id="shadow", stage=StrategyStage.SHADOW),
    )
    registry.set_champion("champion")

    opportunity_risk = BasicOpportunityRiskPolicy()
    portfolio = FixedAllocationPortfolioPolicy(
        allocation_pct=0.02,
        max_open_positions=15,
        max_gross_exposure_pct=0.30,
    )
    portfolio_risk = PassThroughPortfolioRiskPolicy()
    engine = TradingEngine(
        candidate_source=source,
        strategy_provider=registry,
        opportunity_risk=opportunity_risk,
        portfolio_policy=portfolio,
        portfolio_risk=portfolio_risk,
        position_manager=HoldPositions(),
        state_provider=state,
        broker=broker,
    )
    shadows = ShadowStrategyEvaluator(
        registry,
        opportunity_risk=opportunity_risk,
        portfolio_policy=portfolio,
        portfolio_risk=portfolio_risk,
    )

    result = TradingService(engine, shadow_evaluator=shadows).run_cycle(_NOW)

    assert source.calls == 1
    assert state.calls == 1
    assert broker.settle_calls == 1
    assert shadow.seen[0][0] is champion.seen[0][0]
    assert shadow.seen[0][1] is champion.seen[0][1]
    assert result.shadows[0].strategy_id == "shadow"
    assert result.shadows[0].allocation_count == 1
    assert result.shadows[0].selected_candidate_ids == ("candidate-a",)
    # Only the champion reaches the broker.
    assert len(broker.orders) == 1
    assert broker.orders[0].strategy_id == "champion"
    assert result.champion.strategy_id == "champion"


def test_shadow_evaluator_ignores_development_and_live_plugins_by_default() -> None:
    registry = StrategyRegistry()
    champion = _Strategy("champion", 0.99)
    shadow = _Strategy("shadow", 0.98)
    development = _Strategy("development", 0.97)
    live_other = _Strategy("live-other", 0.96)
    for strategy, stage in (
        (champion, StrategyStage.PAPER),
        (shadow, StrategyStage.SHADOW),
        (development, StrategyStage.DEVELOPMENT),
        (live_other, StrategyStage.LIVE),
    ):
        registry.register(
            strategy,
            StrategyRecord(strategy_id=strategy.strategy_id, stage=stage),
        )
    registry.set_champion("champion")
    prepared_portfolio = PortfolioSnapshot(
        as_of=_NOW,
        equity=10_000.0,
        cash=10_000.0,
        gross_exposure_pct=0.0,
    )
    source = _Source()
    from stock_trading.engine import PreparedEngineCycle

    prepared = PreparedEngineCycle(
        as_of=_NOW,
        portfolio=prepared_portfolio,
        candidates=source.batch,
    )
    evaluator = ShadowStrategyEvaluator(
        registry,
        opportunity_risk=BasicOpportunityRiskPolicy(),
        portfolio_policy=FixedAllocationPortfolioPolicy(),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
    )

    results = evaluator.evaluate(prepared)

    assert [item.strategy_id for item in results] == ["shadow"]
    assert len(shadow.seen) == 1
    assert development.seen == []
    assert live_other.seen == []
