from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from stock_trading.market.labels import build_standard_labels
from stock_trading.market.models import MarketBar
from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.multi_horizon import build_multi_horizon_targets, row_for_horizon


class _FakeMarketStore:
    def __init__(self, stock_bars, benchmark_bars) -> None:
        self.stock_bars = stock_bars
        self.benchmark_bars = benchmark_bars

    def security_for_company(self, company_id, day):
        return "stock"

    def bars_from(self, security_id, start_day, limit):
        bars = self.benchmark_bars if security_id == "benchmark" else self.stock_bars
        return [bar for bar in bars if bar.date >= start_day][:limit]


def _bars(security_id: str, slope: Decimal) -> list[MarketBar]:
    start = date(2024, 1, 2)
    result = []
    for offset in range(60):
        price = Decimal("100") + slope * Decimal(offset)
        result.append(
            MarketBar(
                security_id=security_id,
                ticker=security_id.upper(),
                date=start + timedelta(days=offset),
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price + Decimal("0.5"),
                volume=Decimal("1000"),
                adj_open=price,
                adj_high=price + Decimal("1"),
                adj_low=price - Decimal("1"),
                adj_close=price + Decimal("0.5"),
                adj_volume=Decimal("1000"),
            )
        )
    return result


def test_reconstructs_and_verifies_existing_twenty_day_target() -> None:
    stock = _bars("stock", Decimal("1"))
    benchmark = _bars("benchmark", Decimal("0.2"))
    labels = {
        item.horizon: item
        for item in build_standard_labels(stock, benchmark, horizons=(5, 20, 60))
    }
    twenty = labels[20]
    row = TrainingRow(
        event_id="event",
        company_id="company",
        decision_time=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
        execution_date=twenty.start_date,
        exit_date_20d=twenty.end_date,
        features={"x": 1.0},
        stock_return_20d=twenty.stock_return,
        benchmark_return_20d=twenty.benchmark_return,
        alpha_20d=twenty.alpha,
        downside_20d=max(0.0, -twenty.max_adverse_excursion),
        mfe_20d=max(0.0, twenty.max_favorable_excursion),
        positive_alpha_20d=int(twenty.alpha >= 0.02),
    )

    targets = build_multi_horizon_targets(
        (row,),
        _FakeMarketStore(stock, benchmark),
        benchmark_security_id="benchmark",
    )

    assert set(targets["event"]) == {5, 20, 60}
    projected = row_for_horizon(row, targets["event"][5])
    assert projected.exit_date_20d == labels[5].end_date
    assert projected.stock_return_20d == labels[5].stock_return
    assert projected.alpha_20d == labels[5].alpha
