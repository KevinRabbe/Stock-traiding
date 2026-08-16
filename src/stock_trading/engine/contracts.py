from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Mapping


FeatureValue = float | int | str | bool | None


class StrategyStage(StrEnum):
    DEVELOPMENT = "development"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"
    RETIRED = "retired"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """One point-in-time candidate presented to a strategy."""

    candidate_id: str
    event_id: str
    company_id: str
    security_id: str
    decision_time: datetime
    execution_date: date
    features: Mapping[str, FeatureValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Opportunity:
    """Strategy output. Strategies rank opportunities but never place orders."""

    strategy_id: str
    candidate_id: str
    event_id: str
    company_id: str
    security_id: str
    execution_date: date
    score: float
    expected_return: float
    expected_alpha: float
    expected_downside: float
    probability_positive: float
    horizon_sessions: int
    metadata: Mapping[str, FeatureValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be > 0")
        if not 0.0 <= self.probability_positive <= 1.0:
            raise ValueError("probability_positive must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    position_id: str
    strategy_id: str
    company_id: str
    security_id: str
    allocation_pct: float
    opened_on: date
    planned_exit_date: date | None = None
    metadata: Mapping[str, FeatureValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allocation_pct <= 0:
            raise ValueError("position allocation_pct must be > 0")


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    as_of: datetime
    equity: float
    cash: float
    gross_exposure_pct: float
    positions: tuple[PortfolioPosition, ...] = ()

    def __post_init__(self) -> None:
        if self.equity < 0 or self.cash < 0:
            raise ValueError("equity and cash must be >= 0")
        if not 0.0 <= self.gross_exposure_pct <= 1.0:
            raise ValueError("gross_exposure_pct must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AllocationIntent:
    opportunity: Opportunity
    allocation_pct: float
    reason: str

    def __post_init__(self) -> None:
        if self.allocation_pct <= 0:
            raise ValueError("allocation_pct must be > 0")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    order_id: str
    strategy_id: str
    company_id: str
    security_id: str
    side: OrderSide
    allocation_pct: float
    created_at: datetime
    candidate_id: str | None = None
    event_id: str | None = None
    horizon_sessions: int | None = None
    execute_on: date | None = None
    reason: str = ""
    metadata: Mapping[str, FeatureValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allocation_pct <= 0:
            raise ValueError("order allocation_pct must be > 0")
        if self.horizon_sessions is not None and self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be > 0 when provided")
        if self.execute_on is not None and self.execute_on < self.created_at.date():
            raise ValueError("execute_on cannot precede order creation date")


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    order_id: str
    accepted: bool
    executed_at: datetime
    message: str = ""
    fill_price: float | None = None
    status: ExecutionStatus = ExecutionStatus.FILLED

    def __post_init__(self) -> None:
        if self.status in {ExecutionStatus.QUEUED, ExecutionStatus.FILLED} and not self.accepted:
            raise ValueError("queued/filled execution reports must be accepted")
        if self.status in {ExecutionStatus.REJECTED, ExecutionStatus.CANCELLED} and self.accepted:
            raise ValueError("rejected/cancelled execution reports cannot be accepted")
        if self.fill_price is not None and self.fill_price <= 0:
            raise ValueError("fill_price must be > 0 when provided")


@dataclass(frozen=True, slots=True)
class EngineCycleResult:
    as_of: datetime
    strategy_id: str
    candidate_count: int
    opportunity_count: int
    eligible_opportunity_count: int
    allocation_count: int
    position_orders: tuple[OrderIntent, ...]
    entry_orders: tuple[OrderIntent, ...]
    executions: tuple[ExecutionReport, ...]
    settlements: tuple[ExecutionReport, ...] = ()
