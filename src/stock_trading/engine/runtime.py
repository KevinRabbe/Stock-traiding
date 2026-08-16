from __future__ import annotations

from hashlib import sha256

from .contracts import (
    AllocationIntent,
    EngineCycleResult,
    FeatureSnapshot,
    Opportunity,
    OrderIntent,
    OrderSide,
)
from .protocols import (
    CandidateSource,
    EngineObserver,
    ExecutionBroker,
    OpportunityRiskPolicy,
    PortfolioPolicy,
    PortfolioRiskPolicy,
    PortfolioStateProvider,
    PositionManager,
    StrategyProvider,
)


class TradingEngine:
    """Strategy-agnostic orchestration for one decision/execution cycle."""

    def __init__(
        self,
        *,
        candidate_source: CandidateSource,
        strategy_provider: StrategyProvider,
        opportunity_risk: OpportunityRiskPolicy,
        portfolio_policy: PortfolioPolicy,
        portfolio_risk: PortfolioRiskPolicy,
        position_manager: PositionManager,
        state_provider: PortfolioStateProvider,
        broker: ExecutionBroker,
        observer: EngineObserver | None = None,
    ) -> None:
        self._candidate_source = candidate_source
        self._strategy_provider = strategy_provider
        self._opportunity_risk = opportunity_risk
        self._portfolio_policy = portfolio_policy
        self._portfolio_risk = portfolio_risk
        self._position_manager = position_manager
        self._state_provider = state_provider
        self._broker = broker
        self._observer = observer

    def run_cycle(self, as_of) -> EngineCycleResult:
        portfolio = self._state_provider.snapshot(as_of)
        if portfolio.as_of != as_of:
            raise ValueError("portfolio snapshot as_of does not match engine cycle")

        position_orders = self._position_manager.orders(portfolio, as_of)
        candidates = self._candidate_source.candidates(as_of)
        candidate_by_id = _unique_candidates(candidates)

        strategy = self._strategy_provider.active()
        opportunities = strategy.evaluate(candidates, portfolio)
        _validate_opportunities(strategy.strategy_id, opportunities, candidate_by_id)

        eligible = self._opportunity_risk.filter(opportunities, portfolio)
        _validate_subset(opportunities, eligible, "opportunity risk")

        proposed_allocations = self._portfolio_policy.allocate(eligible, portfolio)
        _validate_allocations(proposed_allocations, eligible)
        allocations = self._portfolio_risk.filter(proposed_allocations, portfolio)
        _validate_allocation_subset(proposed_allocations, allocations)

        entry_orders = tuple(_entry_order(item, as_of) for item in allocations)
        all_orders = (*position_orders, *entry_orders)
        _require_unique_order_ids(all_orders)
        executions = self._broker.execute(tuple(all_orders))
        _validate_execution_reports(all_orders, executions)

        result = EngineCycleResult(
            as_of=as_of,
            strategy_id=strategy.strategy_id,
            candidate_count=len(candidates),
            opportunity_count=len(opportunities),
            eligible_opportunity_count=len(eligible),
            allocation_count=len(allocations),
            position_orders=position_orders,
            entry_orders=entry_orders,
            executions=executions,
        )
        if self._observer is not None:
            self._observer.record(result)
        return result


def _unique_candidates(
    candidates: tuple[FeatureSnapshot, ...],
) -> dict[str, FeatureSnapshot]:
    result: dict[str, FeatureSnapshot] = {}
    for candidate in candidates:
        if candidate.candidate_id in result:
            raise ValueError(f"duplicate candidate_id {candidate.candidate_id}")
        result[candidate.candidate_id] = candidate
    return result


def _validate_opportunities(
    strategy_id: str,
    opportunities: tuple[Opportunity, ...],
    candidates: dict[str, FeatureSnapshot],
) -> None:
    seen: set[str] = set()
    for opportunity in opportunities:
        if opportunity.strategy_id != strategy_id:
            raise ValueError("strategy emitted opportunity with foreign strategy_id")
        if opportunity.candidate_id in seen:
            raise ValueError(f"strategy emitted duplicate candidate {opportunity.candidate_id}")
        seen.add(opportunity.candidate_id)
        candidate = candidates.get(opportunity.candidate_id)
        if candidate is None:
            raise ValueError(
                f"strategy emitted unknown candidate {opportunity.candidate_id}"
            )
        if (
            opportunity.event_id != candidate.event_id
            or opportunity.company_id != candidate.company_id
            or opportunity.security_id != candidate.security_id
            or opportunity.execution_date != candidate.execution_date
        ):
            raise ValueError(
                f"strategy changed candidate identity for {opportunity.candidate_id}"
            )


def _validate_subset(
    source: tuple[Opportunity, ...],
    subset: tuple[Opportunity, ...],
    label: str,
) -> None:
    allowed = {item.candidate_id for item in source}
    returned = [item.candidate_id for item in subset]
    if len(returned) != len(set(returned)):
        raise ValueError(f"{label} returned duplicate opportunities")
    unknown = set(returned) - allowed
    if unknown:
        raise ValueError(f"{label} introduced unknown opportunities: {sorted(unknown)}")


def _validate_allocations(
    allocations: tuple[AllocationIntent, ...],
    opportunities: tuple[Opportunity, ...],
) -> None:
    allowed = {item.candidate_id: item for item in opportunities}
    seen: set[str] = set()
    for allocation in allocations:
        candidate_id = allocation.opportunity.candidate_id
        if candidate_id in seen:
            raise ValueError(f"portfolio policy duplicated candidate {candidate_id}")
        seen.add(candidate_id)
        expected = allowed.get(candidate_id)
        if expected is None or expected != allocation.opportunity:
            raise ValueError("portfolio policy introduced or mutated an opportunity")


def _validate_allocation_subset(
    source: tuple[AllocationIntent, ...],
    subset: tuple[AllocationIntent, ...],
) -> None:
    allowed = {item.opportunity.candidate_id: item for item in source}
    seen: set[str] = set()
    for allocation in subset:
        candidate_id = allocation.opportunity.candidate_id
        if candidate_id in seen:
            raise ValueError(f"portfolio risk duplicated candidate {candidate_id}")
        seen.add(candidate_id)
        expected = allowed.get(candidate_id)
        if expected is None:
            raise ValueError("portfolio risk introduced an allocation")
        if allocation.opportunity != expected.opportunity:
            raise ValueError("portfolio risk mutated opportunity identity")
        if allocation.allocation_pct > expected.allocation_pct + 1e-15:
            raise ValueError("portfolio risk may reduce/reject but not upsize allocation")


def _entry_order(allocation: AllocationIntent, as_of) -> OrderIntent:
    opportunity = allocation.opportunity
    digest = sha256(
        f"{opportunity.strategy_id}|{opportunity.candidate_id}|buy".encode("utf-8")
    ).hexdigest()[:20]
    return OrderIntent(
        order_id=f"ord_{digest}",
        strategy_id=opportunity.strategy_id,
        candidate_id=opportunity.candidate_id,
        event_id=opportunity.event_id,
        company_id=opportunity.company_id,
        security_id=opportunity.security_id,
        side=OrderSide.BUY,
        allocation_pct=allocation.allocation_pct,
        created_at=as_of,
        horizon_sessions=opportunity.horizon_sessions,
        reason=allocation.reason,
        metadata={
            "score": opportunity.score,
            "expected_return": opportunity.expected_return,
            "expected_alpha": opportunity.expected_alpha,
            "expected_downside": opportunity.expected_downside,
            "probability_positive": opportunity.probability_positive,
            **dict(opportunity.metadata),
        },
    )


def _require_unique_order_ids(orders: tuple[OrderIntent, ...]) -> None:
    ids = [order.order_id for order in orders]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate order_id in engine cycle")


def _validate_execution_reports(
    orders: tuple[OrderIntent, ...],
    executions,
) -> None:
    expected = {order.order_id for order in orders}
    returned = [report.order_id for report in executions]
    if len(returned) != len(set(returned)):
        raise ValueError("broker returned duplicate execution report")
    if set(returned) != expected:
        raise ValueError("broker must return exactly one execution report per order")
