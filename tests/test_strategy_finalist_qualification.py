from __future__ import annotations

from datetime import date
from dataclasses import replace

import pytest

from stock_trading.research import HistoricalYearResult
from stock_trading.research.historical import (
    HistoricalBacktestResult,
    HistoricalTrade,
)
from stock_trading.research.strategy_factory import StrategyVariantResult, generate_population
from stock_trading.experiments.lightgbm_strategy_qualify import (
    _assert_screening_identity,
    _concentration_diagnostics,
    _spec_from_json,
)


def _trade(
    candidate_id: str,
    company_id: str,
    *,
    pnl: float,
    gross_return: float,
    alpha: float = 0.01,
    horizon: int = 20,
) -> HistoricalTrade:
    return HistoricalTrade(
        strategy_id="strategy",
        candidate_id=candidate_id,
        event_id=candidate_id,
        company_id=company_id,
        security_id=f"sec-{company_id}",
        entry_date=date(2020, 1, 2),
        exit_date=date(2020, 2, 3),
        horizon_sessions=horizon,
        allocation_pct=0.02,
        allocated_capital=200.0,
        gross_return=gross_return,
        net_return=gross_return - 0.002,
        alpha=alpha,
        downside=0.02,
        pnl=pnl,
        opportunity_score=0.95,
    )


def _year(year: int, total_return: float, trades: tuple[HistoricalTrade, ...]) -> HistoricalYearResult:
    positive = sum(item.pnl for item in trades if item.pnl > 0)
    negative = -sum(item.pnl for item in trades if item.pnl < 0)
    pf = positive / negative if negative else (float("inf") if positive else 0.0)
    return HistoricalYearResult(
        year,
        HistoricalBacktestResult(
            starting_capital=10_000.0,
            ending_capital=10_000.0 * (1.0 + total_return),
            total_return=total_return,
            profit_factor=pf,
            realized_max_drawdown=0.02,
            trades=trades,
            rejected_cash=0,
        ),
    )


def test_concentration_diagnostics_exposes_trade_company_and_year_dependence() -> None:
    years = (
        _year(
            2020,
            0.20,
            (
                _trade("a", "company-a", pnl=60.0, gross_return=0.30),
                _trade("b", "company-a", pnl=20.0, gross_return=0.10),
                _trade("c", "company-b", pnl=-10.0, gross_return=-0.05),
            ),
        ),
        _year(
            2021,
            0.05,
            (
                _trade("d", "company-c", pnl=20.0, gross_return=0.10),
                _trade("e", "company-d", pnl=10.0, gross_return=0.05),
            ),
        ),
        _year(
            2022,
            -0.02,
            (_trade("f", "company-e", pnl=-10.0, gross_return=-0.05),),
        ),
        _year(
            2023,
            0.01,
            (_trade("g", "company-f", pnl=5.0, gross_return=0.025),),
        ),
    )

    diagnostics = _concentration_diagnostics(years)

    assert diagnostics["trade_count"] == 7
    assert diagnostics["unique_company_count"] == 6
    assert diagnostics["largest_company_trade_count"] == 2
    assert diagnostics["largest_positive_trade_pnl_fraction"] == pytest.approx(60 / 115)
    assert diagnostics["largest_positive_company_pnl_fraction"] == pytest.approx(80 / 115)
    assert diagnostics["best_year"] == 2020
    assert diagnostics["worst_year"] == 2022
    assert diagnostics["best_three_years"] == [2020, 2021, 2023]
    assert diagnostics["compounded_return_excluding_best_three_years"] == pytest.approx(-0.02)
    assert diagnostics["gross_return_distribution"]["max"] == pytest.approx(0.30)


def test_screening_identity_accepts_exact_result_and_rejects_trade_drift() -> None:
    spec = replace(generate_population(population_size=1)[0], variant_id="candidate")
    result = StrategyVariantResult(
        spec=spec,
        compounded_return=0.10,
        profit_factor=1.5,
        worst_realized_drawdown=0.02,
        total_trades=2,
        profitable_year_rate=0.75,
        average_trade_alpha=0.01,
        compounded_return_excluding_best_year=0.04,
        best_year=2020,
        yearly_returns={2020: 0.06, 2021: 0.04},
        trade_candidate_ids=("a", "b"),
        trade_horizon_counts={20: 2},
    )
    screening = result.as_json()

    _assert_screening_identity(screening, result, tolerance=1e-12)

    changed = dict(screening)
    changed["trade_candidate_ids"] = ["a", "c"]
    with pytest.raises(ValueError, match="trade identity changed"):
        _assert_screening_identity(changed, result, tolerance=1e-12)


def test_spec_round_trip_from_factory_json() -> None:
    spec = generate_population(generation_seed=3, population_size=1)[0]
    restored = _spec_from_json(spec.as_json())
    assert restored == spec
