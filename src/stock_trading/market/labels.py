from collections.abc import Sequence
from dataclasses import dataclass

from .models import MarketBar


@dataclass(frozen=True, slots=True)
class ForwardLabel:
    horizon: int
    stock_return: float
    benchmark_return: float
    alpha: float
    max_favorable_excursion: float
    max_adverse_excursion: float


def build_forward_label(
    stock_bars: Sequence[MarketBar],
    benchmark_bars: Sequence[MarketBar],
    *,
    horizon: int,
) -> ForwardLabel:
    """Build an open-to-close forward label over aligned trading sessions."""

    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    stock_by_date = {bar.date: bar for bar in stock_bars}
    benchmark_by_date = {bar.date: bar for bar in benchmark_bars}
    common_dates = sorted(set(stock_by_date) & set(benchmark_by_date))
    if len(common_dates) < horizon:
        raise ValueError("not enough aligned future trading bars")

    dates = common_dates[:horizon]
    first_stock = stock_by_date[dates[0]]
    first_benchmark = benchmark_by_date[dates[0]]
    last_stock = stock_by_date[dates[-1]]
    last_benchmark = benchmark_by_date[dates[-1]]

    stock_entry = float(first_stock.adj_open)
    benchmark_entry = float(first_benchmark.adj_open)
    stock_return = float(last_stock.adj_close) / stock_entry - 1.0
    benchmark_return = float(last_benchmark.adj_close) / benchmark_entry - 1.0

    maximum = max(float(stock_by_date[day].adj_high) / stock_entry - 1.0 for day in dates)
    minimum = min(float(stock_by_date[day].adj_low) / stock_entry - 1.0 for day in dates)

    return ForwardLabel(
        horizon=horizon,
        stock_return=stock_return,
        benchmark_return=benchmark_return,
        alpha=stock_return - benchmark_return,
        max_favorable_excursion=maximum,
        max_adverse_excursion=minimum,
    )


def build_standard_labels(
    stock_bars: Sequence[MarketBar],
    benchmark_bars: Sequence[MarketBar],
    horizons: tuple[int, ...] = (1, 5, 20, 60),
) -> tuple[ForwardLabel, ...]:
    return tuple(
        build_forward_label(stock_bars, benchmark_bars, horizon=horizon)
        for horizon in horizons
        if len(stock_bars) >= horizon and len(benchmark_bars) >= horizon
    )
