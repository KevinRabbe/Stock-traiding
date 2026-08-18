from __future__ import annotations

from datetime import datetime

from stock_trading.engine import OrderIntent, OrderSide

from .paper import PaperExecutionBroker, PaperPositionState


class SessionClosePaperExecutionBroker(PaperExecutionBroker):
    """Daily-bar paper broker that fills queued orders on their intended session.

    The base broker queues an order until a price becomes available. When a process
    resumes on a later calendar day, using the resume timestamp for a daily-close
    price would silently move the fill to that later day. This adapter instead asks
    the PriceProvider for the order's ``execute_on`` date and records the effective
    fill/open timestamp on that session date while retaining the resume time-of-day
    and timezone. It is intended for the current daily-bar PAPER runtime.
    """

    def _fill(
        self,
        order: OrderIntent,
        as_of: datetime,
        cash: float,
        positions: list[PaperPositionState],
    ):
        execute_on = order.execute_on or order.created_at.date()
        if execute_on > as_of.date():
            return None
        effective_at = datetime.combine(execute_on, as_of.timetz())
        price = self.price_provider.price(order.security_id, effective_at)
        if price is None:
            return None
        price = float(price)
        if price <= 0:
            raise ValueError("price provider returned non-positive price")
        if order.side is OrderSide.BUY:
            return self._fill_buy(order, effective_at, price, cash, positions)
        return self._fill_sell(order, effective_at, price, cash, positions)
