from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from stock_trading.engine import PortfolioPosition, PortfolioSnapshot
from stock_trading.positions import FixedHorizonPositionManager


class _MarketStore:
    def __init__(self, dates):
        self.dates = tuple(dates)
        self.calls = []

    def bars_before(self, security_id, cutoff, limit):
        self.calls.append((security_id, cutoff, limit))
        return [SimpleNamespace(date=day) for day in self.dates if day < cutoff][-limit:]


def _portfolio(as_of, *, horizon=5):
    return PortfolioSnapshot(
        as_of=as_of,
        equity=10_000.0,
        cash=9_800.0,
        gross_exposure_pct=0.02,
        positions=(
            PortfolioPosition(
                position_id="position-a",
                strategy_id="v5",
                company_id="company-a",
                security_id="security-a",
                allocation_pct=0.02,
                opened_on=date(2025, 1, 2),
                metadata={"horizon_sessions": horizon},
            ),
        ),
    )


def test_fixed_horizon_manager_exits_only_after_observed_sessions() -> None:
    store = _MarketStore(
        (
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
        )
    )
    manager = FixedHorizonPositionManager(store)
    before = datetime(2025, 1, 7, 20, tzinfo=timezone.utc)
    due = datetime(2025, 1, 8, 20, tzinfo=timezone.utc)

    assert manager.orders(_portfolio(before), before, (), ()) == ()
    orders = manager.orders(_portfolio(due), due, (), ())

    assert len(orders) == 1
    order = orders[0]
    assert order.company_id == "company-a"
    assert order.execute_on == due.date()
    assert order.reason == "strategy_horizon_complete"
    assert order.metadata["held_sessions"] == 5
    # Query cutoff is only tomorrow relative to the current cycle, never a future
    # horizon-derived date from the historical database.
    assert store.calls[-1][1] == date(2025, 1, 9)


def test_fixed_horizon_exit_order_id_is_stable_across_retry() -> None:
    store = _MarketStore(tuple(date(2025, 1, day) for day in range(2, 10)))
    manager = FixedHorizonPositionManager(store)
    as_of = datetime(2025, 1, 8, 20, tzinfo=timezone.utc)

    first = manager.orders(_portfolio(as_of), as_of, (), ())[0]
    second = manager.orders(_portfolio(as_of), as_of, (), ())[0]

    assert first.order_id == second.order_id


def test_fixed_horizon_manager_rejects_invalid_position_horizon() -> None:
    store = _MarketStore((date(2025, 1, 2),))
    manager = FixedHorizonPositionManager(store)
    as_of = datetime(2025, 1, 2, 20, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="invalid horizon_sessions"):
        manager.orders(_portfolio(as_of, horizon=0), as_of, (), ())
