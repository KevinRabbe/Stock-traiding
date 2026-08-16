from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from stock_trading.ml.dataset import TrainingRow
from stock_trading.research.strategy_factory import (
    PopulationGate,
    StrategyVariantResult,
    apply_feature_profile,
    design_space_size,
    generate_population,
    select_diverse_finalists,
    trade_overlap,
    training_window_rows,
)


_NOW = datetime(2025, 1, 2, tzinfo=timezone.utc)


def _row(year: int) -> TrainingRow:
    return TrainingRow(
        event_id=f"event-{year}",
        company_id="company",
        decision_time=datetime(year, 6, 1, tzinfo=timezone.utc),
        execution_date=date(year, 6, 2),
        exit_date_20d=date(year, 7, 1),
        features={
            "market.return_20d": 0.1,
            "system.regime.benchmark_positive_20d": 1.0,
            "trigger.is_contract": 1.0,
            "contracts.surprise_30d": 2.0,
            "opportunity_history.count_30d": 3.0,
            "unclassified.experimental": 99.0,
        },
        stock_return_20d=0.04,
        benchmark_return_20d=0.01,
        alpha_20d=0.03,
        downside_20d=0.02,
        mfe_20d=0.08,
        positive_alpha_20d=1,
    )


def _result(
    variant_id: str,
    *,
    compound: float,
    pf: float,
    drawdown: float,
    trades: int,
    profitable_rate: float,
    excluding_best: float,
    trade_ids: tuple[str, ...],
) -> StrategyVariantResult:
    spec = replace(generate_population(population_size=1)[0], variant_id=variant_id)
    return StrategyVariantResult(
        spec=spec,
        compounded_return=compound,
        profit_factor=pf,
        worst_realized_drawdown=drawdown,
        total_trades=trades,
        profitable_year_rate=profitable_rate,
        average_trade_alpha=0.01,
        compounded_return_excluding_best_year=excluding_best,
        best_year=2020,
        yearly_returns={2020: 0.02, 2021: 0.01},
        trade_candidate_ids=trade_ids,
        trade_horizon_counts={20: trades},
    )


def test_population_generation_is_deterministic_unique_and_sampled_from_large_grid() -> None:
    first = generate_population(generation_seed=7, population_size=48)
    second = generate_population(generation_seed=7, population_size=48)
    different = generate_population(generation_seed=8, population_size=48)

    assert first == second
    assert len(first) == 48
    assert len({item.variant_id for item in first}) == 48
    assert first != different
    assert design_space_size() > 48


def test_feature_profiles_do_not_change_labels_or_identity() -> None:
    row = _row(2024)
    market = apply_feature_profile((row,), "market_regime")[0]
    event = apply_feature_profile((row,), "event_history")[0]

    assert set(market.features) == {
        "market.return_20d",
        "system.regime.benchmark_positive_20d",
    }
    assert set(event.features) == {
        "trigger.is_contract",
        "contracts.surprise_30d",
        "opportunity_history.count_30d",
    }
    for projected in (market, event):
        assert projected.event_id == row.event_id
        assert projected.company_id == row.company_id
        assert projected.stock_return_20d == row.stock_return_20d
        assert projected.alpha_20d == row.alpha_20d


def test_training_window_uses_only_recent_pre_validation_years() -> None:
    rows = tuple(_row(year) for year in range(2012, 2020))
    selected = training_window_rows(rows, test_year=2020, window_years=5)

    assert [row.decision_time.year for row in selected] == [2014, 2015, 2016, 2017, 2018, 2019]
    # annual_walk_forward has already removed the validation year before calling
    # this helper; prove the lower-bound behavior independently here.
    pre_validation = tuple(row for row in selected if row.decision_time.year < 2019)
    assert [row.decision_time.year for row in pre_validation] == [2014, 2015, 2016, 2017, 2018]


def test_trade_overlap_uses_opportunity_identity_not_horizon() -> None:
    left = _result(
        "left",
        compound=0.10,
        pf=1.5,
        drawdown=0.02,
        trades=100,
        profitable_rate=0.6,
        excluding_best=0.05,
        trade_ids=("a", "b", "c"),
    )
    right = _result(
        "right",
        compound=0.08,
        pf=1.4,
        drawdown=0.02,
        trades=100,
        profitable_rate=0.6,
        excluding_best=0.04,
        trade_ids=("b", "c", "d"),
    )

    assert trade_overlap(left, right) == pytest.approx(0.5)


def test_finalist_selection_keeps_profitable_diversity_instead_of_near_duplicate() -> None:
    best = _result(
        "best",
        compound=0.12,
        pf=1.7,
        drawdown=0.02,
        trades=120,
        profitable_rate=0.7,
        excluding_best=0.08,
        trade_ids=("a", "b", "c", "d"),
    )
    duplicate = _result(
        "duplicate",
        compound=0.11,
        pf=1.65,
        drawdown=0.021,
        trades=115,
        profitable_rate=0.69,
        excluding_best=0.075,
        trade_ids=("a", "b", "c", "d", "x"),
    )
    diverse = _result(
        "diverse",
        compound=0.08,
        pf=1.4,
        drawdown=0.018,
        trades=100,
        profitable_rate=0.65,
        excluding_best=0.06,
        trade_ids=("m", "n", "o", "p"),
    )
    loser = _result(
        "loser",
        compound=-0.01,
        pf=0.9,
        drawdown=0.02,
        trades=150,
        profitable_rate=0.4,
        excluding_best=-0.03,
        trade_ids=("z",),
    )

    selection = select_diverse_finalists(
        (best, duplicate, diverse, loser),
        gate=PopulationGate(min_trades=75),
        finalist_count=2,
        max_trade_overlap=0.70,
    )

    ids = [item.variant_id for item in selection.finalists]
    assert ids[0] == "best"
    assert "diverse" in ids
    assert "duplicate" not in ids
    assert "loser" in selection.rejected_gate
