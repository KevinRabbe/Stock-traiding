from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.engine import (
    ExecutionReport,
    ExecutionStatus,
    OrderIntent,
    OrderSide,
    PortfolioPosition,
    PortfolioSnapshot,
)

from .prices import PriceProvider


@dataclass(frozen=True, slots=True)
class PaperPositionState:
    position_id: str
    strategy_id: str
    company_id: str
    security_id: str
    shares: float
    average_entry_price: float
    opened_at: datetime
    last_price: float
    horizon_sessions: int | None = None
    candidate_id: str | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        if self.shares <= 0:
            raise ValueError("paper position shares must be > 0")
        if self.average_entry_price <= 0 or self.last_price <= 0:
            raise ValueError("paper position prices must be > 0")


@dataclass(frozen=True, slots=True)
class PaperLedgerState:
    cash: float
    positions: tuple[PaperPositionState, ...] = ()
    pending_orders: tuple[OrderIntent, ...] = ()
    completed_reports: tuple[ExecutionReport, ...] = ()
    submitted_orders: tuple[OrderIntent, ...] = ()

    def __post_init__(self) -> None:
        if self.cash < -1e-9:
            raise ValueError("paper cash must be >= 0")
        position_ids = [item.position_id for item in self.positions]
        if len(position_ids) != len(set(position_ids)):
            raise ValueError("duplicate paper position_id")
        pending_ids = [item.order_id for item in self.pending_orders]
        if len(pending_ids) != len(set(pending_ids)):
            raise ValueError("duplicate pending paper order_id")
        completed_ids = [item.order_id for item in self.completed_reports]
        if len(completed_ids) != len(set(completed_ids)):
            raise ValueError("duplicate completed paper order_id")
        if set(pending_ids) & set(completed_ids):
            raise ValueError("paper order cannot be pending and completed")
        submitted_ids = [item.order_id for item in self.submitted_orders]
        if len(submitted_ids) != len(set(submitted_ids)):
            raise ValueError("duplicate submitted paper order_id")


class FilePaperLedger:
    """Atomic durable paper account state."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, *, starting_cash: float = 10_000.0) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be > 0")
        self.path = Path(path)
        self.starting_cash = float(starting_cash)

    def load(self) -> PaperLedgerState:
        if not self.path.exists():
            return PaperLedgerState(cash=self.starting_cash)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid paper ledger at {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported paper ledger schema")
        try:
            pending_orders = tuple(
                _order_from_json(item) for item in payload.get("pending_orders", ())
            )
            raw_submitted = payload.get("submitted_orders")
            submitted_orders = (
                tuple(_order_from_json(item) for item in raw_submitted)
                if raw_submitted is not None
                else pending_orders
            )
            return PaperLedgerState(
                cash=float(payload["cash"]),
                positions=tuple(_position_from_json(item) for item in payload.get("positions", ())),
                pending_orders=pending_orders,
                completed_reports=tuple(
                    _report_from_json(item) for item in payload.get("completed_reports", ())
                ),
                submitted_orders=submitted_orders,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid paper ledger at {self.path}") from exc

    def save(self, state: PaperLedgerState) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "cash": state.cash,
            "positions": [_position_to_json(item) for item in state.positions],
            "pending_orders": [_order_to_json(item) for item in state.pending_orders],
            "completed_reports": [_report_to_json(item) for item in state.completed_reports],
            "submitted_orders": [_order_to_json(item) for item in state.submitted_orders],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class PaperExecutionBroker:
    """Persistent idempotent paper broker with future-date order queuing."""

    def __init__(
        self,
        ledger: FilePaperLedger,
        price_provider: PriceProvider,
        *,
        per_side_cost_bps: float = 10.0,
    ) -> None:
        if per_side_cost_bps < 0:
            raise ValueError("per_side_cost_bps must be >= 0")
        self.ledger = ledger
        self.price_provider = price_provider
        self.per_side_cost_bps = float(per_side_cost_bps)

    def execute(self, orders: tuple[OrderIntent, ...]) -> tuple[ExecutionReport, ...]:
        if not orders:
            return ()
        state = self.ledger.load()
        positions = list(state.positions)
        pending = {order.order_id: order for order in state.pending_orders}
        completed = {report.order_id: report for report in state.completed_reports}
        submitted = {order.order_id: order for order in state.submitted_orders}
        for pending_order in state.pending_orders:
            submitted.setdefault(pending_order.order_id, pending_order)
        cash = state.cash
        reports: list[ExecutionReport] = []

        for order in orders:
            previous_order = submitted.get(order.order_id)
            if previous_order is not None:
                _validate_replayed_order(previous_order, order)
            else:
                submitted[order.order_id] = order

            previous = completed.get(order.order_id)
            if previous is not None:
                reports.append(previous)
                continue
            if order.order_id in pending:
                reports.append(_queued_report(order, order.created_at, "already queued"))
                continue

            execute_on = order.execute_on or order.created_at.date()
            if execute_on > order.created_at.date():
                pending[order.order_id] = order
                reports.append(_queued_report(order, order.created_at, "waiting for execution date"))
                continue

            filled = self._fill(order, order.created_at, cash, positions)
            if filled is None:
                pending[order.order_id] = order
                reports.append(_queued_report(order, order.created_at, "waiting for market price"))
                continue
            cash, final_report = filled
            completed[order.order_id] = final_report
            reports.append(final_report)

        self.ledger.save(
            PaperLedgerState(
                cash=cash,
                positions=tuple(positions),
                pending_orders=tuple(sorted(pending.values(), key=_order_sort_key)),
                completed_reports=tuple(sorted(completed.values(), key=lambda item: item.order_id)),
                submitted_orders=tuple(sorted(submitted.values(), key=_order_sort_key)),
            )
        )
        return tuple(reports)

    def settle(self, as_of: datetime) -> tuple[ExecutionReport, ...]:
        state = self.ledger.load()
        if not state.pending_orders:
            return ()
        positions = list(state.positions)
        pending = {order.order_id: order for order in state.pending_orders}
        completed = {report.order_id: report for report in state.completed_reports}
        submitted = {order.order_id: order for order in state.submitted_orders}
        for pending_order in state.pending_orders:
            submitted.setdefault(pending_order.order_id, pending_order)
        cash = state.cash
        reports: list[ExecutionReport] = []

        for order in sorted(state.pending_orders, key=_order_sort_key):
            execute_on = order.execute_on or order.created_at.date()
            if execute_on > as_of.date():
                continue
            filled = self._fill(order, as_of, cash, positions)
            if filled is None:
                continue
            cash, final_report = filled
            pending.pop(order.order_id, None)
            completed[order.order_id] = final_report
            reports.append(final_report)

        self.ledger.save(
            PaperLedgerState(
                cash=cash,
                positions=tuple(positions),
                pending_orders=tuple(sorted(pending.values(), key=_order_sort_key)),
                completed_reports=tuple(sorted(completed.values(), key=lambda item: item.order_id)),
                submitted_orders=tuple(sorted(submitted.values(), key=_order_sort_key)),
            )
        )
        return tuple(reports)

    def _fill(
        self,
        order: OrderIntent,
        as_of: datetime,
        cash: float,
        positions: list[PaperPositionState],
    ) -> tuple[float, ExecutionReport] | None:
        price = self.price_provider.price(order.security_id, as_of)
        if price is None:
            return None
        price = float(price)
        if price <= 0:
            raise ValueError("price provider returned non-positive price")
        if order.side is OrderSide.BUY:
            return self._fill_buy(order, as_of, price, cash, positions)
        return self._fill_sell(order, as_of, price, cash, positions)

    def _fill_buy(
        self,
        order: OrderIntent,
        as_of: datetime,
        price: float,
        cash: float,
        positions: list[PaperPositionState],
        *,
        mark_price_provider: PriceProvider | None = None,
    ) -> tuple[float, ExecutionReport]:
        if any(item.company_id == order.company_id for item in positions):
            return cash, _rejected_report(order, as_of, "company already has an open position")

        marks = mark_price_provider or self.price_provider
        equity = _marked_equity(cash, positions, marks, as_of)
        target_notional = equity * order.allocation_pct
        fee_rate = self.per_side_cost_bps / 10_000.0
        max_notional = cash / (1.0 + fee_rate) if fee_rate >= 0 else cash
        notional = min(target_notional, max_notional)
        if notional <= 1e-12:
            return cash, _rejected_report(order, as_of, "insufficient cash")
        fee = notional * fee_rate
        shares = notional / price
        cash -= notional + fee
        digest = sha256(
            f"paper|{order.strategy_id}|{order.order_id}|{order.company_id}".encode("utf-8")
        ).hexdigest()[:20]
        positions.append(
            PaperPositionState(
                position_id=f"ppos_{digest}",
                strategy_id=order.strategy_id,
                company_id=order.company_id,
                security_id=order.security_id,
                shares=shares,
                average_entry_price=price,
                opened_at=as_of,
                last_price=price,
                horizon_sessions=order.horizon_sessions,
                candidate_id=order.candidate_id,
                event_id=order.event_id,
            )
        )
        return cash, ExecutionReport(
            order_id=order.order_id,
            accepted=True,
            executed_at=as_of,
            message=f"paper buy filled; fee={fee:.8f}",
            fill_price=price,
            status=ExecutionStatus.FILLED,
        )

    def _fill_sell(
        self,
        order: OrderIntent,
        as_of: datetime,
        price: float,
        cash: float,
        positions: list[PaperPositionState],
        *,
        mark_price_provider: PriceProvider | None = None,
    ) -> tuple[float, ExecutionReport]:
        match_index = next(
            (
                index
                for index, item in enumerate(positions)
                if item.company_id == order.company_id
                and item.security_id == order.security_id
                and item.strategy_id == order.strategy_id
            ),
            None,
        )
        if match_index is None:
            return cash, _rejected_report(order, as_of, "position is not open")
        position = positions[match_index]
        marks = mark_price_provider or self.price_provider
        equity = _marked_equity(cash, positions, marks, as_of)
        position_value = position.shares * price
        if bool(order.metadata.get("full_exit")):
            notional = position_value
        else:
            target_notional = equity * order.allocation_pct
            notional = min(position_value, target_notional)
            if target_notional >= position_value * (1.0 - 1e-9):
                notional = position_value
        shares_to_sell = min(position.shares, notional / price)
        if shares_to_sell <= 1e-12:
            return cash, _rejected_report(order, as_of, "sell amount is too small")
        gross_proceeds = shares_to_sell * price
        fee = gross_proceeds * self.per_side_cost_bps / 10_000.0
        cash += gross_proceeds - fee
        remaining = position.shares - shares_to_sell
        if remaining <= max(1e-12, position.shares * 1e-9):
            positions.pop(match_index)
        else:
            positions[match_index] = PaperPositionState(
                position_id=position.position_id,
                strategy_id=position.strategy_id,
                company_id=position.company_id,
                security_id=position.security_id,
                shares=remaining,
                average_entry_price=position.average_entry_price,
                opened_at=position.opened_at,
                last_price=price,
                horizon_sessions=position.horizon_sessions,
                candidate_id=position.candidate_id,
                event_id=position.event_id,
            )
        return cash, ExecutionReport(
            order_id=order.order_id,
            accepted=True,
            executed_at=as_of,
            message=f"paper sell filled; fee={fee:.8f}",
            fill_price=price,
            status=ExecutionStatus.FILLED,
        )


class PaperPortfolioStateProvider:
    """Mark the durable paper ledger into the generic engine portfolio snapshot."""

    def __init__(self, ledger: FilePaperLedger, price_provider: PriceProvider) -> None:
        self.ledger = ledger
        self.price_provider = price_provider

    def snapshot(self, as_of: datetime) -> PortfolioSnapshot:
        state = self.ledger.load()
        values: list[tuple[PaperPositionState, float, float]] = []
        for position in state.positions:
            current_price = self.price_provider.price(position.security_id, as_of)
            mark = float(current_price) if current_price is not None else position.last_price
            if mark <= 0:
                raise ValueError("paper position has no valid mark price")
            values.append((position, mark, position.shares * mark))
        gross_value = sum(value for _, _, value in values)
        equity = state.cash + gross_value
        gross_exposure = gross_value / equity if equity > 0 else 0.0
        positions = tuple(
            PortfolioPosition(
                position_id=position.position_id,
                strategy_id=position.strategy_id,
                company_id=position.company_id,
                security_id=position.security_id,
                allocation_pct=value / equity if equity > 0 else 0.0,
                opened_on=position.opened_at.date(),
                metadata={
                    "shares": position.shares,
                    "average_entry_price": position.average_entry_price,
                    "mark_price": mark,
                    "horizon_sessions": position.horizon_sessions,
                    "candidate_id": position.candidate_id,
                    "event_id": position.event_id,
                },
            )
            for position, mark, value in values
            if value > 0
        )
        return PortfolioSnapshot(
            as_of=as_of,
            equity=equity,
            cash=max(0.0, state.cash),
            gross_exposure_pct=min(1.0, max(0.0, gross_exposure)),
            positions=positions,
        )


def _marked_equity(
    cash: float,
    positions: list[PaperPositionState],
    prices: PriceProvider,
    as_of: datetime,
) -> float:
    equity = cash
    for position in positions:
        if position.opened_at.date() == as_of.date():
            price = position.last_price
        else:
            current = prices.price(position.security_id, as_of)
            price = float(current) if current is not None else position.last_price
        equity += position.shares * price
    return equity


def _queued_report(order: OrderIntent, at: datetime, message: str) -> ExecutionReport:
    return ExecutionReport(
        order_id=order.order_id,
        accepted=True,
        executed_at=at,
        message=message,
        status=ExecutionStatus.QUEUED,
    )


def _rejected_report(order: OrderIntent, at: datetime, message: str) -> ExecutionReport:
    return ExecutionReport(
        order_id=order.order_id,
        accepted=False,
        executed_at=at,
        message=message,
        status=ExecutionStatus.REJECTED,
    )


def _order_sort_key(order: OrderIntent):
    return (order.execute_on or order.created_at.date(), order.created_at, order.order_id)


def _validate_replayed_order(previous: OrderIntent, current: OrderIntent) -> None:
    """Reject reuse of a deterministic order ID for a different economic intent.

    ``created_at`` is intentionally excluded: an idempotent process retry has a new
    wall-clock timestamp while the durable order identity must remain unchanged.
    Metadata is also excluded because diagnostic/model annotations are not execution
    authority; the economic order contract below is.
    """

    previous_identity = (
        previous.strategy_id,
        previous.candidate_id,
        previous.event_id,
        previous.company_id,
        previous.security_id,
        previous.side,
        previous.allocation_pct,
        previous.horizon_sessions,
        previous.execute_on,
    )
    current_identity = (
        current.strategy_id,
        current.candidate_id,
        current.event_id,
        current.company_id,
        current.security_id,
        current.side,
        current.allocation_pct,
        current.horizon_sessions,
        current.execute_on,
    )
    if previous_identity != current_identity:
        raise ValueError("paper order_id was replayed with different economic intent")


def _position_to_json(position: PaperPositionState) -> dict:
    return {
        "position_id": position.position_id,
        "strategy_id": position.strategy_id,
        "company_id": position.company_id,
        "security_id": position.security_id,
        "shares": position.shares,
        "average_entry_price": position.average_entry_price,
        "opened_at": position.opened_at.isoformat(),
        "last_price": position.last_price,
        "horizon_sessions": position.horizon_sessions,
        "candidate_id": position.candidate_id,
        "event_id": position.event_id,
    }


def _position_from_json(payload: dict) -> PaperPositionState:
    return PaperPositionState(
        position_id=str(payload["position_id"]),
        strategy_id=str(payload["strategy_id"]),
        company_id=str(payload["company_id"]),
        security_id=str(payload["security_id"]),
        shares=float(payload["shares"]),
        average_entry_price=float(payload["average_entry_price"]),
        opened_at=datetime.fromisoformat(payload["opened_at"]),
        last_price=float(payload["last_price"]),
        horizon_sessions=(
            int(payload["horizon_sessions"])
            if payload.get("horizon_sessions") is not None
            else None
        ),
        candidate_id=payload.get("candidate_id"),
        event_id=payload.get("event_id"),
    )


def _order_to_json(order: OrderIntent) -> dict:
    return {
        "order_id": order.order_id,
        "strategy_id": order.strategy_id,
        "company_id": order.company_id,
        "security_id": order.security_id,
        "side": order.side.value,
        "allocation_pct": order.allocation_pct,
        "created_at": order.created_at.isoformat(),
        "candidate_id": order.candidate_id,
        "event_id": order.event_id,
        "horizon_sessions": order.horizon_sessions,
        "execute_on": order.execute_on.isoformat() if order.execute_on is not None else None,
        "reason": order.reason,
        "metadata": dict(order.metadata),
    }


def _order_from_json(payload: dict) -> OrderIntent:
    return OrderIntent(
        order_id=str(payload["order_id"]),
        strategy_id=str(payload["strategy_id"]),
        company_id=str(payload["company_id"]),
        security_id=str(payload["security_id"]),
        side=OrderSide(str(payload["side"])),
        allocation_pct=float(payload["allocation_pct"]),
        created_at=datetime.fromisoformat(payload["created_at"]),
        candidate_id=payload.get("candidate_id"),
        event_id=payload.get("event_id"),
        horizon_sessions=(
            int(payload["horizon_sessions"])
            if payload.get("horizon_sessions") is not None
            else None
        ),
        execute_on=(
            date.fromisoformat(payload["execute_on"])
            if payload.get("execute_on") is not None
            else None
        ),
        reason=str(payload.get("reason") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def _report_to_json(report: ExecutionReport) -> dict:
    return {
        "order_id": report.order_id,
        "accepted": report.accepted,
        "executed_at": report.executed_at.isoformat(),
        "message": report.message,
        "fill_price": report.fill_price,
        "status": report.status.value,
    }


def _report_from_json(payload: dict) -> ExecutionReport:
    return ExecutionReport(
        order_id=str(payload["order_id"]),
        accepted=bool(payload["accepted"]),
        executed_at=datetime.fromisoformat(payload["executed_at"]),
        message=str(payload.get("message") or ""),
        fill_price=(float(payload["fill_price"]) if payload.get("fill_price") is not None else None),
        status=ExecutionStatus(str(payload.get("status") or ExecutionStatus.FILLED.value)),
    )
