from __future__ import annotations

from stock_trading.engine import OrderIntent, OrderSide


def receipt_champion_entry_order_ids(
    *,
    batch_id: str,
    champion_strategy_id: str,
    emitted_entry_orders: tuple[OrderIntent, ...],
    broker,
) -> tuple[str, ...]:
    """Recover the exact durable champion BUY set for one runtime batch.

    The PAPER broker journals full submitted orders atomically with broker state and
    tags authoritative runtime submissions with ``runtime_batch_id``. A retry after
    broker durability but before receipt publication may emit no new entry because
    the queued order already reserves portfolio capacity. The journal remains the
    authoritative way to reconstruct the receipt without guessing by company/date.
    """

    if not batch_id.strip() or not champion_strategy_id.strip():
        raise ValueError("batch and champion strategy identity must not be empty")
    provider = getattr(broker, "submitted_entry_orders", None)
    if not callable(provider):
        raise TypeError("PAPER broker does not expose submitted entry order journal")

    durable_orders = tuple(
        order
        for order in provider()
        if order.side is OrderSide.BUY
        and order.strategy_id == champion_strategy_id
        and order.metadata.get("runtime_batch_id") == batch_id
    )
    durable_ids = {order.order_id for order in durable_orders}
    emitted_ids = {order.order_id for order in emitted_entry_orders}
    missing = sorted(emitted_ids - durable_ids)
    if missing:
        raise RuntimeError(
            "champion entry orders are missing from durable submitted PAPER journal: "
            f"{missing}"
        )
    return tuple(sorted(durable_ids))
