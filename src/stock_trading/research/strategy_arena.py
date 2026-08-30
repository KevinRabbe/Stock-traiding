from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import inf, isfinite, isnan
from typing import Mapping, Sequence


class StrategyLifecycle(str, Enum):
    """Research lifecycle for strategies competing in the arena."""

    EXPERIMENTAL = "experimental"
    SHADOW = "shadow"
    ACTIVE = "active"
    DORMANT = "dormant"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class MarketStateSnapshot:
    """Point-in-time description of the market environment.

    The arena deliberately does not define how these dimensions are produced. A
    simple rules engine, clustering model, HMM, neural encoder, or exchange-specific
    state detector can all feed the same boundary later.
    """

    as_of: datetime
    regime_id: str
    confidence: float
    dimensions: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.regime_id.strip():
            raise ValueError("regime_id must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("market-state confidence must be in [0, 1]")
        for name, value in self.dimensions.items():
            if not str(name).strip():
                raise ValueError("market-state dimension names must not be empty")
            if not isfinite(float(value)):
                raise ValueError("market-state dimensions must be finite")


@dataclass(frozen=True, slots=True)
class StrategyObservation:
    """Current research evidence for one strategy under the current state.

    ``expectancy`` and ``recent_expectancy`` are net of modeled trading costs.
    ``regime_similarity`` describes how strongly the present state matches the
    historical states from which the strategy's conditional evidence was measured.
    ``stability`` is a normalized robustness input supplied by the experiment layer.

    ``invalidated`` is reserved for structural/data failures. Ordinary bad market
    fit should make a strategy dormant rather than permanently deleting it.
    """

    strategy_id: str
    family: str
    trade_count: int
    expectancy: float
    recent_expectancy: float
    profit_factor: float
    max_drawdown: float
    regime_similarity: float
    stability: float
    average_alpha: float | None = None
    invalidated: bool = False

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must not be empty")
        if not self.family.strip():
            raise ValueError("strategy family must not be empty")
        if self.trade_count < 0:
            raise ValueError("trade_count must be >= 0")
        for name, value in (
            ("expectancy", self.expectancy),
            ("recent_expectancy", self.recent_expectancy),
            ("max_drawdown", self.max_drawdown),
            ("regime_similarity", self.regime_similarity),
            ("stability", self.stability),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.average_alpha is not None and not isfinite(float(self.average_alpha)):
            raise ValueError("average_alpha must be finite or None")
        if self.profit_factor < 0 or isnan(float(self.profit_factor)):
            raise ValueError("profit_factor must be >= 0 and not NaN")
        if self.max_drawdown < 0:
            raise ValueError("max_drawdown must be >= 0")
        for name, value in (
            ("regime_similarity", self.regime_similarity),
            ("stability", self.stability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ArenaPolicy:
    """Research-only promotion and allocation policy.

    These defaults are intentionally conservative and have no PAPER/live authority.
    A separate immutable portfolio-risk layer remains responsible for real capital.
    """

    min_shadow_trades: int = 20
    min_active_trades: int = 75
    min_profit_factor: float = 1.05
    min_expectancy: float = 0.0
    min_recent_expectancy: float = 0.0
    max_drawdown: float = 0.05
    min_regime_similarity: float = 0.60
    min_market_confidence: float = 0.55
    max_active_strategies: int = 8
    max_strategy_weight: float = 0.25
    max_total_weight: float = 0.80

    def __post_init__(self) -> None:
        if self.min_shadow_trades < 0:
            raise ValueError("min_shadow_trades must be >= 0")
        if self.min_active_trades < self.min_shadow_trades:
            raise ValueError("min_active_trades must be >= min_shadow_trades")
        if self.min_profit_factor < 0:
            raise ValueError("min_profit_factor must be >= 0")
        if self.max_drawdown < 0:
            raise ValueError("max_drawdown must be >= 0")
        for name, value in (
            ("min_regime_similarity", self.min_regime_similarity),
            ("min_market_confidence", self.min_market_confidence),
            ("max_strategy_weight", self.max_strategy_weight),
            ("max_total_weight", self.max_total_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_active_strategies <= 0:
            raise ValueError("max_active_strategies must be > 0")
        if self.max_strategy_weight <= 0 or self.max_total_weight <= 0:
            raise ValueError("allocation limits must be > 0")


@dataclass(frozen=True, slots=True)
class ArenaStrategyState:
    strategy_id: str
    family: str
    lifecycle: StrategyLifecycle
    fitness: float
    active_eligible: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ArenaAllocation:
    """Research allocation proposal, never an order or broker instruction."""

    strategy_id: str
    weight: float
    fitness: float


@dataclass(frozen=True, slots=True)
class ArenaDecision:
    market_state: MarketStateSnapshot
    strategies: tuple[ArenaStrategyState, ...]
    allocations: tuple[ArenaAllocation, ...]
    cash_reserve_weight: float


class StrategyArena:
    """Select strategies for the current state while preserving survival discipline.

    The arena sits above strategy generation/backtesting and below the immutable risk
    and execution layers. It does not place trades, change strategy code, or alter
    broker/risk constraints.
    """

    def __init__(self, policy: ArenaPolicy | None = None) -> None:
        self.policy = policy or ArenaPolicy()

    def evaluate(
        self,
        market_state: MarketStateSnapshot,
        observations: Sequence[StrategyObservation],
        *,
        previous_states: Mapping[str, StrategyLifecycle] | None = None,
    ) -> ArenaDecision:
        materialized = tuple(observations)
        ids = [item.strategy_id for item in materialized]
        if len(ids) != len(set(ids)):
            raise ValueError("strategy arena received duplicate strategy_id values")

        previous = dict(previous_states or {})
        fitness = _composite_fitness(materialized)
        states: list[ArenaStrategyState] = []
        for observation in materialized:
            prior = previous.get(observation.strategy_id, StrategyLifecycle.EXPERIMENTAL)
            eligible = _active_eligible(observation, market_state, self.policy)
            lifecycle = _next_lifecycle(
                prior,
                observation,
                active_eligible=eligible,
                policy=self.policy,
            )
            states.append(
                ArenaStrategyState(
                    strategy_id=observation.strategy_id,
                    family=observation.family,
                    lifecycle=lifecycle,
                    fitness=fitness.get(observation.strategy_id, 0.0),
                    active_eligible=eligible,
                    reason=_state_reason(
                        lifecycle,
                        observation,
                        market_state,
                        self.policy,
                    ),
                )
            )

        active = sorted(
            (item for item in states if item.lifecycle is StrategyLifecycle.ACTIVE),
            key=lambda item: (-item.fitness, item.strategy_id),
        )[: self.policy.max_active_strategies]
        weights = _capped_proportional_weights(
            [(item.strategy_id, item.fitness) for item in active],
            max_each=self.policy.max_strategy_weight,
            max_total=self.policy.max_total_weight,
        )
        allocations = tuple(
            ArenaAllocation(
                strategy_id=item.strategy_id,
                weight=weights[item.strategy_id],
                fitness=item.fitness,
            )
            for item in active
            if weights.get(item.strategy_id, 0.0) > 0
        )
        allocated = sum(item.weight for item in allocations)
        return ArenaDecision(
            market_state=market_state,
            strategies=tuple(sorted(states, key=lambda item: item.strategy_id)),
            allocations=allocations,
            cash_reserve_weight=max(0.0, 1.0 - allocated),
        )


def _active_eligible(
    observation: StrategyObservation,
    market_state: MarketStateSnapshot,
    policy: ArenaPolicy,
) -> bool:
    return (
        not observation.invalidated
        and observation.trade_count >= policy.min_active_trades
        and observation.expectancy > policy.min_expectancy
        and observation.recent_expectancy > policy.min_recent_expectancy
        and observation.profit_factor >= policy.min_profit_factor
        and observation.max_drawdown <= policy.max_drawdown
        and observation.regime_similarity >= policy.min_regime_similarity
        and market_state.confidence >= policy.min_market_confidence
    )


def _next_lifecycle(
    previous: StrategyLifecycle,
    observation: StrategyObservation,
    *,
    active_eligible: bool,
    policy: ArenaPolicy,
) -> StrategyLifecycle:
    if previous is StrategyLifecycle.RETIRED or observation.invalidated:
        return StrategyLifecycle.RETIRED
    if observation.trade_count < policy.min_shadow_trades:
        return StrategyLifecycle.EXPERIMENTAL
    if previous is StrategyLifecycle.EXPERIMENTAL:
        return StrategyLifecycle.SHADOW
    if active_eligible:
        return StrategyLifecycle.ACTIVE
    if previous in (StrategyLifecycle.ACTIVE, StrategyLifecycle.DORMANT):
        return StrategyLifecycle.DORMANT
    return StrategyLifecycle.SHADOW


def _state_reason(
    lifecycle: StrategyLifecycle,
    observation: StrategyObservation,
    market_state: MarketStateSnapshot,
    policy: ArenaPolicy,
) -> str:
    if lifecycle is StrategyLifecycle.RETIRED:
        return "structural_or_data_invalidation"
    if lifecycle is StrategyLifecycle.EXPERIMENTAL:
        return "insufficient_shadow_evidence"
    if lifecycle is StrategyLifecycle.ACTIVE:
        return "positive_expectancy_and_current_regime_fit"
    failures: list[str] = []
    if observation.trade_count < policy.min_active_trades:
        failures.append("sample")
    if observation.expectancy <= policy.min_expectancy:
        failures.append("expectancy")
    if observation.recent_expectancy <= policy.min_recent_expectancy:
        failures.append("recent_expectancy")
    if observation.profit_factor < policy.min_profit_factor:
        failures.append("profit_factor")
    if observation.max_drawdown > policy.max_drawdown:
        failures.append("drawdown")
    if observation.regime_similarity < policy.min_regime_similarity:
        failures.append("regime_fit")
    if market_state.confidence < policy.min_market_confidence:
        failures.append("market_state_confidence")
    if lifecycle is StrategyLifecycle.DORMANT:
        return "dormant:" + ",".join(failures or ["active_gate"])
    return "shadow:" + ",".join(failures or ["awaiting_promotion_cycle"])


def _composite_fitness(
    observations: Sequence[StrategyObservation],
) -> dict[str, float]:
    usable = [item for item in observations if not item.invalidated]
    if not usable:
        return {}
    expectancy = _percentile_ranks({item.strategy_id: item.expectancy for item in usable})
    recent = _percentile_ranks(
        {item.strategy_id: item.recent_expectancy for item in usable}
    )
    profit_factor = _percentile_ranks(
        {
            item.strategy_id: (
                item.profit_factor if isfinite(item.profit_factor) else 1e9
            )
            for item in usable
        }
    )
    drawdown = _percentile_ranks(
        {item.strategy_id: -item.max_drawdown for item in usable}
    )
    regime = _percentile_ranks(
        {item.strategy_id: item.regime_similarity for item in usable}
    )
    stability = _percentile_ranks(
        {item.strategy_id: item.stability for item in usable}
    )
    return {
        item.strategy_id: (
            0.25 * expectancy[item.strategy_id]
            + 0.20 * recent[item.strategy_id]
            + 0.15 * profit_factor[item.strategy_id]
            + 0.15 * drawdown[item.strategy_id]
            + 0.15 * regime[item.strategy_id]
            + 0.10 * stability[item.strategy_id]
        )
        for item in usable
    }


def _percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    return {
        key: index / (len(ordered) - 1)
        for index, (key, _) in enumerate(ordered)
    }


def _capped_proportional_weights(
    ranked: Sequence[tuple[str, float]],
    *,
    max_each: float,
    max_total: float,
) -> dict[str, float]:
    if not ranked:
        return {}
    if max_each <= 0 or max_total <= 0:
        return {strategy_id: 0.0 for strategy_id, _ in ranked}

    target = min(max_total, len(ranked) * max_each)
    remaining_ids = [strategy_id for strategy_id, _ in ranked]
    strength = {
        strategy_id: max(0.05, float(score))
        for strategy_id, score in ranked
    }
    weights = {strategy_id: 0.0 for strategy_id, _ in ranked}
    remaining = target

    while remaining_ids and remaining > 1e-12:
        total_strength = sum(strength[strategy_id] for strategy_id in remaining_ids)
        if total_strength <= 0:
            equal = remaining / len(remaining_ids)
            for strategy_id in remaining_ids:
                weights[strategy_id] = min(max_each, equal)
            break

        provisional = {
            strategy_id: remaining * strength[strategy_id] / total_strength
            for strategy_id in remaining_ids
        }
        capped = [
            strategy_id
            for strategy_id in remaining_ids
            if provisional[strategy_id] > max_each + 1e-12
        ]
        if not capped:
            for strategy_id in remaining_ids:
                weights[strategy_id] = provisional[strategy_id]
            remaining = 0.0
            break

        for strategy_id in capped:
            weights[strategy_id] = max_each
            remaining -= max_each
            remaining_ids.remove(strategy_id)

    return weights
