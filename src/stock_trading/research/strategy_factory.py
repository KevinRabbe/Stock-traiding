from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Iterable, Mapping, Sequence

from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.lightgbm_models import LightGbmTrainingConfig


FEATURE_PROFILES: dict[str, tuple[str, ...] | None] = {
    "full": None,
    "market_regime": (
        "market.",
        "system.regime.",
        "system.momentum.",
        "system.volatility.",
        "system.cross_section.",
    ),
    "event_history": (
        "trigger.",
        "insider.",
        "contracts.",
        "lobbying.",
        "congress.",
        "cross.",
        "interaction.",
        "opportunity_history.",
        "system.interaction.",
    ),
    "balanced_core": (
        "market.",
        "trigger.",
        "insider.",
        "contracts.",
        "lobbying.",
        "congress.",
        "cross.",
        "interaction.",
        "opportunity_history.",
        "system.",
    ),
}


@dataclass(frozen=True, slots=True)
class TreeProfile:
    name: str
    num_boost_round: int
    early_stopping_rounds: int
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float

    def training_config(self, *, seed: int) -> LightGbmTrainingConfig:
        return LightGbmTrainingConfig(
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=1,
            downside_penalty=0.5,
            balance_companies=True,
            seed=seed,
        )


TREE_PROFILES: dict[str, TreeProfile] = {
    "conservative": TreeProfile(
        name="conservative",
        num_boost_round=700,
        early_stopping_rounds=60,
        learning_rate=0.02,
        num_leaves=15,
        min_data_in_leaf=40,
        feature_fraction=0.80,
        bagging_fraction=0.80,
    ),
    "baseline": TreeProfile(
        name="baseline",
        num_boost_round=500,
        early_stopping_rounds=50,
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.90,
        bagging_fraction=0.90,
    ),
    "expressive": TreeProfile(
        name="expressive",
        num_boost_round=400,
        early_stopping_rounds=40,
        learning_rate=0.04,
        num_leaves=63,
        min_data_in_leaf=15,
        feature_fraction=1.00,
        bagging_fraction=0.90,
    ),
}


@dataclass(frozen=True, slots=True)
class StrategyVariantSpec:
    variant_id: str
    feature_profile: str
    training_window_years: int | None
    tree_profile: str
    horizons: tuple[int, ...]
    alpha_rank_weight: float
    seed: int
    validation_top_fraction: float = 0.05
    calibration_window_days: int = 365
    max_expected_downside: float = 0.06

    def __post_init__(self) -> None:
        if self.feature_profile not in FEATURE_PROFILES:
            raise ValueError(f"unknown feature profile {self.feature_profile}")
        if self.tree_profile not in TREE_PROFILES:
            raise ValueError(f"unknown tree profile {self.tree_profile}")
        if self.training_window_years is not None and self.training_window_years <= 0:
            raise ValueError("training_window_years must be positive or None")
        if not self.horizons or any(item <= 0 for item in self.horizons):
            raise ValueError("horizons must contain positive session counts")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if not 0.0 <= self.alpha_rank_weight <= 1.0:
            raise ValueError("alpha_rank_weight must be in [0, 1]")
        if not 0.0 < self.validation_top_fraction < 1.0:
            raise ValueError("validation_top_fraction must be in (0, 1)")
        if self.calibration_window_days <= 0:
            raise ValueError("calibration_window_days must be > 0")
        if self.max_expected_downside < 0:
            raise ValueError("max_expected_downside must be >= 0")

    @property
    def training_config(self) -> LightGbmTrainingConfig:
        return TREE_PROFILES[self.tree_profile].training_config(seed=self.seed)

    def as_json(self) -> dict:
        value = asdict(self)
        value["horizons"] = list(self.horizons)
        return value


@dataclass(frozen=True, slots=True)
class StrategyVariantResult:
    spec: StrategyVariantSpec
    compounded_return: float
    profit_factor: float
    worst_realized_drawdown: float
    total_trades: int
    profitable_year_rate: float
    average_trade_alpha: float | None
    compounded_return_excluding_best_year: float | None
    best_year: int | None
    yearly_returns: Mapping[int, float]
    trade_candidate_ids: tuple[str, ...]
    trade_horizon_counts: Mapping[int, int]

    def as_json(self) -> dict:
        return {
            "spec": self.spec.as_json(),
            "scorecard": {
                "compounded_return": self.compounded_return,
                "profit_factor": self.profit_factor,
                "worst_realized_drawdown": self.worst_realized_drawdown,
                "total_trades": self.total_trades,
                "profitable_year_rate": self.profitable_year_rate,
                "average_trade_alpha": self.average_trade_alpha,
                "compounded_return_excluding_best_year": (
                    self.compounded_return_excluding_best_year
                ),
                "best_year": self.best_year,
            },
            "yearly_returns": {
                str(year): value for year, value in sorted(self.yearly_returns.items())
            },
            "trade_horizon_counts": {
                str(horizon): count
                for horizon, count in sorted(self.trade_horizon_counts.items())
            },
            "trade_candidate_ids": list(self.trade_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class FinalistSelection:
    variant_id: str
    selection_score: float
    maximum_overlap_with_earlier_finalist: float


@dataclass(frozen=True, slots=True)
class PopulationSelection:
    eligible_count: int
    finalists: tuple[FinalistSelection, ...]
    rejected_gate: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PopulationGate:
    min_compounded_return: float = 0.0
    min_profit_factor: float = 1.05
    min_trades: int = 75
    max_realized_drawdown: float = 0.05

    def eligible(self, result: StrategyVariantResult) -> bool:
        return (
            result.compounded_return > self.min_compounded_return
            and result.profit_factor >= self.min_profit_factor
            and result.total_trades >= self.min_trades
            and result.worst_realized_drawdown <= self.max_realized_drawdown
        )


def generate_population(
    *,
    generation_seed: int = 20260816,
    population_size: int = 48,
) -> tuple[StrategyVariantSpec, ...]:
    """Generate a deterministic, constrained structural strategy population.

    The design grid intentionally uses named, interpretable choices instead of a
    continuous hyperparameter optimizer. ``population_size`` samples that grid
    deterministically and the full hypothesis count can be recorded by callers.
    """

    if population_size <= 0:
        raise ValueError("population_size must be > 0")

    feature_profiles = tuple(sorted(FEATURE_PROFILES))
    training_windows = (None, 5, 8)
    tree_profiles = tuple(sorted(TREE_PROFILES))
    horizon_profiles = ((5, 20, 60), (5, 20), (20, 60))
    alpha_weights = (0.0, 0.25, 0.50)
    seeds = (42, 137)

    candidates: list[StrategyVariantSpec] = []
    for feature_profile in feature_profiles:
        for training_window in training_windows:
            for tree_profile in tree_profiles:
                for horizons in horizon_profiles:
                    for alpha_weight in alpha_weights:
                        for seed in seeds:
                            payload = {
                                "feature_profile": feature_profile,
                                "training_window_years": training_window,
                                "tree_profile": tree_profile,
                                "horizons": horizons,
                                "alpha_rank_weight": alpha_weight,
                                "seed": seed,
                            }
                            digest = hashlib.sha256(
                                json.dumps(payload, sort_keys=True).encode("utf-8")
                            ).hexdigest()[:12]
                            candidates.append(
                                StrategyVariantSpec(
                                    variant_id=f"factory-{digest}",
                                    feature_profile=feature_profile,
                                    training_window_years=training_window,
                                    tree_profile=tree_profile,
                                    horizons=horizons,
                                    alpha_rank_weight=alpha_weight,
                                    seed=seed,
                                )
                            )

    candidates.sort(key=lambda item: item.variant_id)
    if population_size >= len(candidates):
        return tuple(candidates)
    rng = random.Random(generation_seed)
    indices = sorted(rng.sample(range(len(candidates)), population_size))
    return tuple(candidates[index] for index in indices)


def design_space_size() -> int:
    return len(generate_population(population_size=10_000))


def apply_feature_profile(
    rows: Iterable[TrainingRow],
    profile: str,
) -> tuple[TrainingRow, ...]:
    prefixes = FEATURE_PROFILES.get(profile)
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"unknown feature profile {profile}")
    if prefixes is None:
        return tuple(rows)

    results: list[TrainingRow] = []
    for row in rows:
        features = {
            name: value
            for name, value in row.features.items()
            if any(name.startswith(prefix) for prefix in prefixes)
        }
        if not features:
            raise ValueError(f"feature profile {profile} produced no features")
        results.append(replace(row, features=features))
    return tuple(results)


def training_window_rows(
    rows: Iterable[TrainingRow],
    *,
    test_year: int,
    window_years: int | None,
) -> tuple[TrainingRow, ...]:
    materialized = tuple(rows)
    if window_years is None:
        return materialized
    if window_years <= 0:
        raise ValueError("window_years must be positive or None")
    validation_year = test_year - 1
    first_year = validation_year - window_years
    return tuple(row for row in materialized if row.decision_time.year >= first_year)


def trade_overlap(left: StrategyVariantResult, right: StrategyVariantResult) -> float:
    left_ids = set(left.trade_candidate_ids)
    right_ids = set(right.trade_candidate_ids)
    union = left_ids | right_ids
    if not union:
        return 0.0
    return len(left_ids & right_ids) / len(union)


def select_diverse_finalists(
    results: Sequence[StrategyVariantResult],
    *,
    gate: PopulationGate | None = None,
    finalist_count: int = 8,
    max_trade_overlap: float = 0.75,
) -> PopulationSelection:
    if finalist_count <= 0:
        raise ValueError("finalist_count must be > 0")
    if not 0.0 <= max_trade_overlap <= 1.0:
        raise ValueError("max_trade_overlap must be in [0, 1]")
    resolved_gate = gate or PopulationGate()
    eligible = [item for item in results if resolved_gate.eligible(item)]
    rejected = tuple(
        sorted(item.spec.variant_id for item in results if not resolved_gate.eligible(item))
    )
    if not eligible:
        return PopulationSelection(0, (), rejected)

    scores = _composite_rank_scores(eligible)
    ordered = sorted(
        eligible,
        key=lambda item: (-scores[item.spec.variant_id], item.spec.variant_id),
    )
    selected: list[StrategyVariantResult] = []
    finalist_rows: list[FinalistSelection] = []
    for candidate in ordered:
        overlaps = [trade_overlap(candidate, other) for other in selected]
        maximum_overlap = max(overlaps, default=0.0)
        if selected and maximum_overlap > max_trade_overlap:
            continue
        selected.append(candidate)
        finalist_rows.append(
            FinalistSelection(
                variant_id=candidate.spec.variant_id,
                selection_score=scores[candidate.spec.variant_id],
                maximum_overlap_with_earlier_finalist=maximum_overlap,
            )
        )
        if len(selected) >= finalist_count:
            break

    return PopulationSelection(
        eligible_count=len(eligible),
        finalists=tuple(finalist_rows),
        rejected_gate=rejected,
    )


def _composite_rank_scores(
    results: Sequence[StrategyVariantResult],
) -> dict[str, float]:
    return_rank = _percentile_ranks(
        {item.spec.variant_id: item.compounded_return for item in results}
    )
    pf_rank = _percentile_ranks(
        {
            item.spec.variant_id: (
                item.profit_factor if isfinite(item.profit_factor) else 1e9
            )
            for item in results
        }
    )
    consistency_rank = _percentile_ranks(
        {item.spec.variant_id: item.profitable_year_rate for item in results}
    )
    drawdown_rank = _percentile_ranks(
        {item.spec.variant_id: -item.worst_realized_drawdown for item in results}
    )
    concentration_rank = _percentile_ranks(
        {
            item.spec.variant_id: (
                item.compounded_return_excluding_best_year
                if item.compounded_return_excluding_best_year is not None
                else -1e9
            )
            for item in results
        }
    )
    return {
        item.spec.variant_id: (
            0.30 * return_rank[item.spec.variant_id]
            + 0.20 * pf_rank[item.spec.variant_id]
            + 0.15 * consistency_rank[item.spec.variant_id]
            + 0.15 * drawdown_rank[item.spec.variant_id]
            + 0.20 * concentration_rank[item.spec.variant_id]
        )
        for item in results
    }


def _percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    return {
        key: index / (len(ordered) - 1)
        for index, (key, _) in enumerate(ordered)
    }
