from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .contracts import AllocationIntent, Opportunity, OrderIntent, PortfolioSnapshot


@dataclass(frozen=True, slots=True)
class FixedAllocationPortfolioPolicy:
    """Simple strategy-agnostic baseline allocation."""

    allocation_pct: float = 0.02
    max_open_positions: int = 15
    max_gross_exposure_pct: float = 0.30
    one_position_per_company: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.allocation_pct <= 1.0:
            raise ValueError("allocation_pct must be in (0, 1]")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        if not 0.0 < self.max_gross_exposure_pct <= 1.0:
            raise ValueError("max_gross_exposure_pct must be in (0, 1]")

    def allocate(
        self,
        opportunities: tuple[Opportunity, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[AllocationIntent, ...]:
        occupied_companies = {position.company_id for position in portfolio.positions}
        available_slots = max(0, self.max_open_positions - len(portfolio.positions))
        remaining_exposure = max(0.0, self.max_gross_exposure_pct - portfolio.gross_exposure_pct)
        results: list[AllocationIntent] = []

        # Match the proven legacy portfolio ordering exactly: highest score first,
        # then stable candidate/event identity ascending for deterministic ties.
        for opportunity in sorted(
            opportunities,
            key=lambda item: (-item.score, item.candidate_id),
        ):
            if len(results) >= available_slots:
                break
            if self.one_position_per_company and opportunity.company_id in occupied_companies:
                continue
            if remaining_exposure + 1e-15 < self.allocation_pct:
                break

            results.append(
                AllocationIntent(
                    opportunity=opportunity,
                    allocation_pct=self.allocation_pct,
                    reason="fixed_allocation_ranked_entry",
                )
            )
            remaining_exposure -= self.allocation_pct
            occupied_companies.add(opportunity.company_id)

        return tuple(results)


@dataclass(frozen=True, slots=True)
class BasicOpportunityRiskPolicy:
    """Shared opportunity-level eligibility independent of strategy implementation."""

    max_expected_downside: float = 0.06
    min_expected_return: float = 0.0
    min_probability_positive: float = 0.0

    def __post_init__(self) -> None:
        if self.max_expected_downside < 0:
            raise ValueError("max_expected_downside must be >= 0")
        if not 0.0 <= self.min_probability_positive <= 1.0:
            raise ValueError("min_probability_positive must be in [0, 1]")

    def filter(
        self,
        opportunities: tuple[Opportunity, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[Opportunity, ...]:
        del portfolio
        return tuple(
            opportunity
            for opportunity in opportunities
            if opportunity.expected_downside <= self.max_expected_downside
            and opportunity.expected_return >= self.min_expected_return
            and opportunity.probability_positive >= self.min_probability_positive
        )


class PassThroughOpportunityRiskPolicy:
    def filter(
        self,
        opportunities: tuple[Opportunity, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[Opportunity, ...]:
        del portfolio
        return opportunities


class PassThroughPortfolioRiskPolicy:
    def filter(
        self,
        allocations: tuple[AllocationIntent, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[AllocationIntent, ...]:
        del portfolio
        return allocations


class HoldPositions:
    """Default position manager until active thesis re-evaluation is plugged in."""

    def orders(
        self,
        portfolio: PortfolioSnapshot,
        as_of: datetime,
    ) -> tuple[OrderIntent, ...]:
        del portfolio, as_of
        return ()


def total_requested_exposure(allocations: Iterable[AllocationIntent]) -> float:
    return sum(item.allocation_pct for item in allocations)
