from __future__ import annotations

from datetime import datetime
from typing import Protocol

from .contracts import (
    AllocationIntent,
    EngineCycleResult,
    ExecutionReport,
    FeatureSnapshot,
    Opportunity,
    OrderIntent,
    PortfolioSnapshot,
)


class CandidateSource(Protocol):
    """Build the point-in-time candidate set available at one engine cycle."""

    def candidates(self, as_of: datetime) -> tuple[FeatureSnapshot, ...]: ...


class OpportunityStrategy(Protocol):
    """Pure opportunity logic. It has no broker or order authority."""

    @property
    def strategy_id(self) -> str: ...

    def evaluate(
        self,
        candidates: tuple[FeatureSnapshot, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[Opportunity, ...]: ...


class StrategyProvider(Protocol):
    """Resolve the currently approved champion strategy."""

    def active(self) -> OpportunityStrategy: ...


class OpportunityRiskPolicy(Protocol):
    """Reject unsafe opportunities before they compete for portfolio capacity."""

    def filter(
        self,
        opportunities: tuple[Opportunity, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[Opportunity, ...]: ...


class PortfolioPolicy(Protocol):
    """Turn ranked, risk-eligible opportunities into desired allocations."""

    def allocate(
        self,
        opportunities: tuple[Opportunity, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[AllocationIntent, ...]: ...


class PortfolioRiskPolicy(Protocol):
    """Apply exposure/correlation/capital rules after allocations are proposed."""

    def filter(
        self,
        allocations: tuple[AllocationIntent, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[AllocationIntent, ...]: ...


class PositionManager(Protocol):
    """Review existing positions using the newest strategy information.

    Position management is deliberately allowed to react to repeat/new signals,
    but this boundary may only reduce/exit existing positions. New or increased
    exposure must continue through opportunity risk + portfolio allocation.
    """

    def orders(
        self,
        portfolio: PortfolioSnapshot,
        as_of: datetime,
        candidates: tuple[FeatureSnapshot, ...],
        opportunities: tuple[Opportunity, ...],
    ) -> tuple[OrderIntent, ...]: ...


class PortfolioStateProvider(Protocol):
    def snapshot(self, as_of: datetime) -> PortfolioSnapshot: ...


class ExecutionBroker(Protocol):
    def execute(self, orders: tuple[OrderIntent, ...]) -> tuple[ExecutionReport, ...]: ...


class EngineObserver(Protocol):
    def record(self, result: EngineCycleResult) -> None: ...
