from __future__ import annotations

from datetime import datetime

from stock_trading.engine import OrderIntent, OrderSide
from stock_trading.market import DuckDbMarketStore

from .paper import FilePaperLedger, PaperExecutionBroker, PaperPositionState
from .prices import (
    DuckDbLatestClosePriceProvider,
    DuckDbPreviousClosePriceProvider,
    PriceProvider,
)


class _PendingEntryReservationMixin:
    """Expose durable queued BUYs to portfolio allocation as reserved capacity."""

    def pending_entry_orders(self) -> tuple[OrderIntent, ...]:
        state = self.ledger.load()
        return tuple(
            order
            for order in state.pending_orders
            if order.side is OrderSide.BUY
        )


class SessionClosePaperExecutionBroker(_PendingEntryReservationMixin, PaperExecutionBroker):
    """Fill queued orders on their intended session instead of restart wall-clock.

    The base paper broker asks its price provider using the settlement timestamp.
    During a restart that could price a Jan-03 order from Jan-05 market data. This
    adapter keeps the durable queue but evaluates a due order against its original
    ``execute_on`` date, matching the candidate's execution session.
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
        return super()._fill(order, effective_at, cash, positions)


class SessionBarPaperExecutionBroker(_PendingEntryReservationMixin, PaperExecutionBroker):
    """Model the strategy's daily-bar execution contract without date drift.

    BUY orders use the adjusted open of their exact ``execute_on`` session because
    current candidates are defined for next-session-open execution. SELL orders use
    the adjusted close of their exact terminal session, matching the forward-label
    convention. The broker may learn those prices later from completed EOD data, but
    a restart never moves an order to the restart day's bar.
    """

    def __init__(
        self,
        ledger: FilePaperLedger,
        market_store: DuckDbMarketStore,
        *,
        per_side_cost_bps: float = 10.0,
    ) -> None:
        self.market_store = market_store
        self.latest_close_provider = DuckDbLatestClosePriceProvider(market_store)
        self.previous_close_provider = DuckDbPreviousClosePriceProvider(market_store)
        super().__init__(
            ledger,
            self.latest_close_provider,
            per_side_cost_bps=per_side_cost_bps,
        )

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
        bar = self.market_store.bar_on(order.security_id, execute_on)
        if bar is None:
            return None

        effective_at = datetime.combine(execute_on, as_of.timetz())
        if order.side is OrderSide.BUY:
            return self._fill_buy(
                order,
                effective_at,
                float(bar.adj_open),
                cash,
                positions,
                mark_price_provider=self.previous_close_provider,
            )
        return self._fill_sell(
            order,
            effective_at,
            float(bar.adj_close),
            cash,
            positions,
            mark_price_provider=self.latest_close_provider,
        )
