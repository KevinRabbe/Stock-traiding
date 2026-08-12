from math import sqrt
from statistics import mean

from .models import MarketBar


_RETURN_HORIZONS = (1, 5, 10, 20, 60, 120)
_VOLATILITY_WINDOWS = (5, 20, 60)


def build_market_features(
    stock_bars: list[MarketBar] | tuple[MarketBar, ...],
    benchmark_bars: list[MarketBar] | tuple[MarketBar, ...],
) -> dict[str, float | None]:
    """Build market-state features from completed, point-in-time-safe daily bars."""

    stock_by_date = {bar.date: bar for bar in stock_bars}
    benchmark_by_date = {bar.date: bar for bar in benchmark_bars}
    common_dates = sorted(set(stock_by_date) & set(benchmark_by_date))
    stock = [stock_by_date[day] for day in common_dates]
    benchmark = [benchmark_by_date[day] for day in common_dates]

    features: dict[str, float | None] = {}
    for horizon in _RETURN_HORIZONS:
        stock_return = _close_return(stock, horizon)
        benchmark_return = _close_return(benchmark, horizon)
        features[f"market.return_{horizon}d"] = stock_return
        features[f"market.benchmark_return_{horizon}d"] = benchmark_return
        features[f"market.relative_return_{horizon}d"] = (
            stock_return - benchmark_return
            if stock_return is not None and benchmark_return is not None
            else None
        )

    return_20d = features["market.return_20d"]
    features["market.appreciation_gt_10pct_20d"] = (
        float(return_20d > 0.10) if return_20d is not None else None
    )

    for window in _VOLATILITY_WINDOWS:
        features[f"market.volatility_{window}d"] = _realized_volatility(stock, window)

    features["market.volume_ratio_5d_20d"] = _volume_ratio(stock, short=5, long=20)
    features["market.volume_zscore_20d"] = _latest_volume_zscore(stock, window=20)
    features["market.avg_dollar_volume_20d"] = _average_dollar_volume(stock, window=20)
    features["market.distance_20d_high"] = _distance_from_high(stock, window=20)
    features["market.distance_20d_low"] = _distance_from_low(stock, window=20)
    features["market.distance_252d_high"] = _distance_from_high(stock, window=252)
    features["market.distance_252d_low"] = _distance_from_low(stock, window=252)
    features["market.within_10pct_252d_high"] = (
        float(features["market.distance_252d_high"] >= -0.10)
        if features["market.distance_252d_high"] is not None
        else None
    )
    return features


def _close_return(bars: list[MarketBar], horizon: int) -> float | None:
    if len(bars) <= horizon:
        return None
    start = float(bars[-horizon - 1].adj_close)
    end = float(bars[-1].adj_close)
    return end / start - 1.0


def _daily_returns(bars: list[MarketBar]) -> list[float]:
    closes = [float(bar.adj_close) for bar in bars]
    return [current / previous - 1.0 for previous, current in zip(closes, closes[1:])]


def _realized_volatility(bars: list[MarketBar], window: int) -> float | None:
    if len(bars) <= window:
        return None
    returns = _daily_returns(bars[-window - 1 :])
    if not returns:
        return None
    avg = mean(returns)
    variance = mean((value - avg) ** 2 for value in returns)
    return sqrt(variance) * sqrt(252.0)


def _volume_ratio(bars: list[MarketBar], *, short: int, long: int) -> float | None:
    if len(bars) < long or short > long:
        return None
    short_average = mean(float(bar.adj_volume) for bar in bars[-short:])
    long_average = mean(float(bar.adj_volume) for bar in bars[-long:])
    if long_average == 0:
        return None
    return short_average / long_average


def _latest_volume_zscore(bars: list[MarketBar], *, window: int) -> float | None:
    if len(bars) <= window:
        return None
    history = [float(bar.adj_volume) for bar in bars[-window - 1 : -1]]
    latest = float(bars[-1].adj_volume)
    average = mean(history)
    variance = mean((value - average) ** 2 for value in history)
    deviation = sqrt(variance)
    if deviation == 0:
        return 0.0
    return (latest - average) / deviation


def _average_dollar_volume(bars: list[MarketBar], *, window: int) -> float | None:
    if len(bars) < window:
        return None
    return mean(float(bar.close) * float(bar.volume) for bar in bars[-window:])


def _distance_from_high(bars: list[MarketBar], *, window: int) -> float | None:
    if len(bars) < window:
        return None
    current = float(bars[-1].adj_close)
    high = max(float(bar.adj_high) for bar in bars[-window:])
    return current / high - 1.0


def _distance_from_low(bars: list[MarketBar], *, window: int) -> float | None:
    if len(bars) < window:
        return None
    current = float(bars[-1].adj_close)
    low = min(float(bar.adj_low) for bar in bars[-window:])
    return current / low - 1.0
