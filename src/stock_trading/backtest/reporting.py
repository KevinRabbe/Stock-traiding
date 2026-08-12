from dataclasses import dataclass
from math import ceil, prod
from statistics import median

from stock_trading.ml.walk_forward import WalkForwardResult

from .portfolio import BacktestResult, ScoredCandidate


@dataclass(frozen=True, slots=True)
class ScoreBucketResult:
    top_fraction: float
    candidate_count: int
    average_alpha_20d: float
    median_alpha_20d: float
    average_stock_return_20d: float
    win_rate_alpha: float
    average_downside_20d: float


@dataclass(frozen=True, slots=True)
class WalkForwardSummary:
    years: tuple[int, ...]
    compounded_return: float
    profitable_year_rate: float
    total_trades: int
    average_trade_alpha: float | None
    aggregate_profit_factor: float | None
    worst_realized_drawdown: float
    worst_trade_mae: float | None


def evaluate_score_buckets(
    candidates: tuple[ScoredCandidate, ...] | list[ScoredCandidate],
    *,
    fractions: tuple[float, ...] = (0.20, 0.10, 0.05, 0.02, 0.01),
) -> tuple[ScoreBucketResult, ...]:
    candidates = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.prediction.opportunity_score,
                candidate.row.event_id,
            ),
        )
    )
    if not candidates:
        return ()

    results: list[ScoreBucketResult] = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("score bucket fractions must be in (0, 1]")
        count = min(len(candidates), max(1, ceil(len(candidates) * fraction)))
        selected = candidates[:count]
        alphas = [candidate.row.alpha_20d for candidate in selected]
        stock_returns = [candidate.row.stock_return_20d for candidate in selected]
        downsides = [candidate.row.downside_20d for candidate in selected]
        results.append(
            ScoreBucketResult(
                top_fraction=fraction,
                candidate_count=count,
                average_alpha_20d=sum(alphas) / count,
                median_alpha_20d=median(alphas),
                average_stock_return_20d=sum(stock_returns) / count,
                win_rate_alpha=sum(alpha > 0 for alpha in alphas) / count,
                average_downside_20d=sum(downsides) / count,
            )
        )
    return tuple(results)


def summarize_walk_forward(
    results: tuple[WalkForwardResult, ...] | list[WalkForwardResult],
) -> WalkForwardSummary:
    results = tuple(results)
    if not results:
        return WalkForwardSummary(
            years=(),
            compounded_return=0.0,
            profitable_year_rate=0.0,
            total_trades=0,
            average_trade_alpha=None,
            aggregate_profit_factor=None,
            worst_realized_drawdown=0.0,
            worst_trade_mae=None,
        )

    trades = [trade for result in results for trade in result.backtest.trades]
    gains = sum(trade.pnl for trade in trades if trade.pnl > 0)
    losses = -sum(trade.pnl for trade in trades if trade.pnl < 0)
    compounded_return = prod(1.0 + result.backtest.total_return for result in results) - 1.0
    trade_alphas = [trade.alpha_20d for trade in trades]
    maes = [trade.max_adverse_excursion for trade in trades]

    return WalkForwardSummary(
        years=tuple(result.test_year for result in results),
        compounded_return=compounded_return,
        profitable_year_rate=sum(result.backtest.net_profit > 0 for result in results) / len(results),
        total_trades=len(trades),
        average_trade_alpha=(sum(trade_alphas) / len(trade_alphas) if trade_alphas else None),
        aggregate_profit_factor=(
            gains / losses if losses > 0 else (float("inf") if gains > 0 else None)
        ),
        worst_realized_drawdown=max(
            result.backtest.realized_max_drawdown for result in results
        ),
        worst_trade_mae=(min(maes) if maes else None),
    )


def profit_without_best_trades(result: BacktestResult, count: int) -> float:
    if count < 0:
        raise ValueError("count must be >= 0")
    if count == 0:
        return result.net_profit
    ordered = sorted(result.trades, key=lambda trade: trade.pnl, reverse=True)
    removed = sum(trade.pnl for trade in ordered[:count])
    return result.net_profit - removed
