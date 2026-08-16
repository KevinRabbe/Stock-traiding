from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from stock_trading.engine import (
    AllocationIntent,
    EngineCycleResult,
    Opportunity,
    PreparedEngineCycle,
    StrategyRegistry,
    StrategyStage,
    TradingEngine,
)
from stock_trading.engine.protocols import (
    OpportunityRiskPolicy,
    PortfolioPolicy,
    PortfolioRiskPolicy,
)
from stock_trading.engine.runtime import validate_strategy_opportunities


@dataclass(frozen=True, slots=True)
class ShadowStrategyResult:
    strategy_id: str
    candidate_count: int
    opportunity_count: int
    eligible_opportunity_count: int
    allocation_count: int
    requested_exposure_pct: float
    top_score: float | None
    horizon_counts: tuple[tuple[int, int], ...]
    selected_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TradingServiceCycle:
    champion: EngineCycleResult
    shadows: tuple[ShadowStrategyResult, ...]


class ShadowObserver(Protocol):
    def record(self, as_of, results: tuple[ShadowStrategyResult, ...]) -> None: ...


class ShadowStrategyEvaluator:
    """Evaluate challengers on the champion's exact PIT context with zero order authority."""

    def __init__(
        self,
        registry: StrategyRegistry,
        *,
        opportunity_risk: OpportunityRiskPolicy,
        portfolio_policy: PortfolioPolicy,
        portfolio_risk: PortfolioRiskPolicy,
        stages: tuple[StrategyStage, ...] = (StrategyStage.SHADOW,),
        observer: ShadowObserver | None = None,
    ) -> None:
        self.registry = registry
        self.opportunity_risk = opportunity_risk
        self.portfolio_policy = portfolio_policy
        self.portfolio_risk = portfolio_risk
        self.stages = stages
        self.observer = observer

    def evaluate(
        self,
        prepared: PreparedEngineCycle,
    ) -> tuple[ShadowStrategyResult, ...]:
        results: list[ShadowStrategyResult] = []
        for strategy in self.registry.loaded_challenger_strategies(stages=self.stages):
            opportunities = strategy.evaluate(prepared.candidates, prepared.portfolio)
            validate_strategy_opportunities(
                strategy.strategy_id,
                opportunities,
                prepared.candidates,
            )
            eligible = self.opportunity_risk.filter(opportunities, prepared.portfolio)
            proposed = self.portfolio_policy.allocate(eligible, prepared.portfolio)
            _validate_shadow_allocations(eligible, proposed)
            allocations = self.portfolio_risk.filter(proposed, prepared.portfolio)
            _validate_shadow_portfolio_risk(proposed, allocations)
            horizon_counts = Counter(item.opportunity.horizon_sessions for item in allocations)
            results.append(
                ShadowStrategyResult(
                    strategy_id=strategy.strategy_id,
                    candidate_count=len(prepared.candidates),
                    opportunity_count=len(opportunities),
                    eligible_opportunity_count=len(eligible),
                    allocation_count=len(allocations),
                    requested_exposure_pct=sum(item.allocation_pct for item in allocations),
                    top_score=max((item.score for item in opportunities), default=None),
                    horizon_counts=tuple(sorted(horizon_counts.items())),
                    selected_candidate_ids=tuple(
                        item.opportunity.candidate_id for item in allocations
                    ),
                )
            )
        resolved = tuple(results)
        if self.observer is not None:
            self.observer.record(prepared.as_of, resolved)
        return resolved


class TradingService:
    """Run shadows and champion from one immutable prepared market/portfolio view."""

    def __init__(
        self,
        engine: TradingEngine,
        *,
        shadow_evaluator: ShadowStrategyEvaluator | None = None,
    ) -> None:
        self.engine = engine
        self.shadow_evaluator = shadow_evaluator

    def run_cycle(self, as_of) -> TradingServiceCycle:
        prepared = self.engine.prepare_cycle(as_of)
        shadows = (
            self.shadow_evaluator.evaluate(prepared)
            if self.shadow_evaluator is not None
            else ()
        )
        champion = self.engine.run_prepared(prepared)
        return TradingServiceCycle(champion=champion, shadows=shadows)


def _validate_shadow_allocations(
    eligible: tuple[Opportunity, ...],
    allocations: tuple[AllocationIntent, ...],
) -> None:
    allowed = {item.candidate_id: item for item in eligible}
    seen: set[str] = set()
    for allocation in allocations:
        candidate_id = allocation.opportunity.candidate_id
        if candidate_id in seen:
            raise ValueError("shadow portfolio produced duplicate candidate")
        seen.add(candidate_id)
        if allowed.get(candidate_id) != allocation.opportunity:
            raise ValueError("shadow portfolio introduced or mutated opportunity")


def _validate_shadow_portfolio_risk(
    proposed: tuple[AllocationIntent, ...],
    filtered: tuple[AllocationIntent, ...],
) -> None:
    allowed = {item.opportunity.candidate_id: item for item in proposed}
    seen: set[str] = set()
    for allocation in filtered:
        candidate_id = allocation.opportunity.candidate_id
        if candidate_id in seen:
            raise ValueError("shadow portfolio risk duplicated candidate")
        seen.add(candidate_id)
        original = allowed.get(candidate_id)
        if original is None or original.opportunity != allocation.opportunity:
            raise ValueError("shadow portfolio risk introduced or mutated opportunity")
        if allocation.allocation_pct > original.allocation_pct + 1e-15:
            raise ValueError("shadow portfolio risk may not upsize allocation")
