from __future__ import annotations

from dataclasses import dataclass
from math import inf, prod

from stock_trading.engine import StrategyScorecard

from .historical import HistoricalBacktestResult


@dataclass(frozen=True, slots=True)
class HistoricalYearResult:
    year: int
    backtest: HistoricalBacktestResult


@dataclass(frozen=True, slots=True)
class HistoricalWalkForwardSummary:
    scorecard: StrategyScorecard
    compounded_return: float
    best_year: int | None
    compounded_return_excluding_best_year: float | None
    year_results: tuple[HistoricalYearResult, ...]


def summarize_historical_years(
    year_results: tuple[HistoricalYearResult, ...],
) -> HistoricalWalkForwardSummary:
    if not year_results:
        raise ValueError("year_results must not be empty")
    years = tuple(sorted(year_results, key=lambda item: item.year))
    compounded = prod(1.0 + item.backtest.total_return for item in years) - 1.0
    profitable_year_rate = (
        sum(item.backtest.total_return > 0 for item in years) / len(years)
    )
    trades = [trade for item in years for trade in item.backtest.trades]
    positive_pnl = sum(trade.pnl for trade in trades if trade.pnl > 0)
    negative_pnl = -sum(trade.pnl for trade in trades if trade.pnl < 0)
    aggregate_profit_factor = (
        positive_pnl / negative_pnl
        if negative_pnl > 0
        else (inf if positive_pnl > 0 else 0.0)
    )
    average_alpha = (
        sum(trade.alpha for trade in trades) / len(trades) if trades else None
    )
    worst_drawdown = max(item.backtest.realized_max_drawdown for item in years)
    scorecard = StrategyScorecard(
        compounded_return=compounded,
        profit_factor=aggregate_profit_factor,
        worst_realized_drawdown=worst_drawdown,
        total_trades=len(trades),
        profitable_year_rate=profitable_year_rate,
        average_trade_alpha=average_alpha,
    )

    best = max(years, key=lambda item: item.backtest.total_return)
    excluding_best = (
        prod(
            1.0 + item.backtest.total_return
            for item in years
            if item.year != best.year
        )
        - 1.0
        if len(years) > 1
        else None
    )
    return HistoricalWalkForwardSummary(
        scorecard=scorecard,
        compounded_return=compounded,
        best_year=best.year,
        compounded_return_excluding_best_year=excluding_best,
        year_results=years,
    )
