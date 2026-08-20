from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable

from stock_trading.engine import OrderIntent, OrderSide
from stock_trading.market import DuckDbMarketStore

from .paper import FilePaperLedger, PaperExecutionBroker, PaperPositionState
from .paper_batch_commit import (
    FilePaperRuntimeBatchCommitStore,
    PaperRuntimeBatchCommit,
)
from .prices import (
    DuckDbLatestClosePriceProvider,
    DuckDbPreviousClosePriceProvider,
    PriceProvider,
)


class _PendingEntryReservationMixin:
    """Expose durable PAPER BUY state for reservation and crash recovery."""

    def pending_entry_orders(self) -> tuple[OrderIntent, ...]:
        state = self.ledger.load()
        return tuple(
            order
            for order in state.pending_orders
            if order.side is OrderSide.BUY
        )

    def submitted_entry_orders(self) -> tuple[OrderIntent, ...]:
        state = self.ledger.load()
        return tuple(
            order
            for order in state.submitted_orders
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

    When ``runtime_batch_id`` is supplied, every submitted order is durably tagged
    with that batch before the atomic ledger save. ``before_runtime_batch_commit``
    runs after all strategy evaluations but before broker durability, so mutable
    calibration state can be persisted before a recoverable PAPER order exists.
    After the broker call succeeds, an atomic batch-commit sidecar is written even
    when no orders were submitted, making zero-order cycles recoverable too.
    """

    def __init__(
        self,
        ledger: FilePaperLedger,
        market_store: DuckDbMarketStore,
        *,
        per_side_cost_bps: float = 10.0,
        runtime_batch_id: str | None = None,
        before_runtime_batch_commit: Callable[[], None] | None = None,
    ) -> None:
        if runtime_batch_id is not None and not runtime_batch_id.strip():
            raise ValueError("runtime_batch_id must not be empty when provided")
        self.market_store = market_store
        self.latest_close_provider = DuckDbLatestClosePriceProvider(market_store)
        self.previous_close_provider = DuckDbPreviousClosePriceProvider(market_store)
        self.runtime_batch_id = runtime_batch_id
        self.before_runtime_batch_commit = before_runtime_batch_commit
        self.runtime_batch_commit_store = (
            FilePaperRuntimeBatchCommitStore.for_ledger(ledger)
            if runtime_batch_id is not None
            else None
        )
        super().__init__(
            ledger,
            self.latest_close_provider,
            per_side_cost_bps=per_side_cost_bps,
        )

    def execute(self, orders: tuple[OrderIntent, ...]):
        if self.runtime_batch_id is None:
            return super().execute(orders)
        existing = {
            order.order_id: order for order in self.ledger.load().submitted_orders
        }
        tagged: list[OrderIntent] = []
        for order in orders:
            previous = existing.get(order.order_id)
            if previous is not None:
                previous_batch = previous.metadata.get("runtime_batch_id")
                if previous_batch is not None and previous_batch != self.runtime_batch_id:
                    raise ValueError(
                        "PAPER order_id belongs to a different runtime batch"
                    )
            tagged.append(_tag_runtime_batch(order, self.runtime_batch_id))
        if self.before_runtime_batch_commit is not None:
            self.before_runtime_batch_commit()
        reports = super().execute(tuple(tagged))
        commit_store = self.runtime_batch_commit_store
        if commit_store is None:
            raise RuntimeError("runtime PAPER batch commit store is not configured")
        commit_store.write(
            PaperRuntimeBatchCommit(
                batch_id=self.runtime_batch_id,
                committed_at=datetime.now(timezone.utc),
                submitted_order_ids=tuple(order.order_id for order in tagged),
            )
        )
        return reports

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


def _tag_runtime_batch(order: OrderIntent, batch_id: str) -> OrderIntent:
    existing = order.metadata.get("runtime_batch_id")
    if existing is not None and existing != batch_id:
        raise ValueError("PAPER order already belongs to a different runtime batch")
    return replace(
        order,
        metadata={**dict(order.metadata), "runtime_batch_id": batch_id},
    )
