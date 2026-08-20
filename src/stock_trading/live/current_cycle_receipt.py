from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from stock_trading.core import as_utc


@dataclass(frozen=True, slots=True)
class CurrentCycleReceipt:
    batch_id: str
    completed_at: datetime
    target_execution_date: date
    selected_event_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    champion_strategy_id: str
    champion_entry_order_ids: tuple[str, ...]
    shadow_strategy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "completed_at", as_utc(self.completed_at))
        if not self.batch_id.strip() or not self.champion_strategy_id.strip():
            raise ValueError("current cycle receipt identity must not be empty")
        if len(self.selected_event_ids) != len(set(self.selected_event_ids)):
            raise ValueError("current cycle receipt contains duplicate selected event IDs")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("current cycle receipt contains duplicate candidate IDs")
        if len(self.champion_entry_order_ids) != len(set(self.champion_entry_order_ids)):
            raise ValueError("current cycle receipt contains duplicate champion entry order IDs")


@dataclass(frozen=True, slots=True)
class ReceiptPaperIntegrityResult:
    receipt_count: int
    referenced_champion_order_count: int
    pending_champion_order_count: int
    completed_champion_order_count: int


@dataclass(frozen=True, slots=True)
class ReceiptReconciliationResult:
    receipt_count: int
    matched_receipt_count: int
    acknowledged_pending_event_count: int
    matched_batch_ids: tuple[str, ...]
    referenced_champion_order_count: int = 0
    pending_champion_order_count: int = 0
    completed_champion_order_count: int = 0


class FileCurrentCycleReceiptStore:
    """One immutable-style atomic receipt per evaluated actionable event batch."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, batch_id: str) -> CurrentCycleReceipt | None:
        path = self._path(batch_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid current cycle receipt: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported current cycle receipt schema")
        try:
            return CurrentCycleReceipt(
                batch_id=str(payload["batch_id"]),
                completed_at=datetime.fromisoformat(str(payload["completed_at"])),
                target_execution_date=date.fromisoformat(
                    str(payload["target_execution_date"])
                ),
                selected_event_ids=tuple(str(item) for item in payload["selected_event_ids"]),
                candidate_ids=tuple(str(item) for item in payload["candidate_ids"]),
                champion_strategy_id=str(payload["champion_strategy_id"]),
                champion_entry_order_ids=tuple(
                    str(item) for item in payload.get("champion_entry_order_ids", ())
                ),
                shadow_strategy_ids=tuple(
                    str(item) for item in payload.get("shadow_strategy_ids", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid current cycle receipt: {path}") from exc

    def load_all(self) -> tuple[CurrentCycleReceipt, ...]:
        if not self.root.exists():
            return ()
        receipts: list[CurrentCycleReceipt] = []
        for path in sorted(self.root.glob("batch_*.json")):
            receipt = self.load(path.stem)
            if receipt is None:
                raise RuntimeError(f"current cycle receipt disappeared during scan: {path}")
            receipts.append(receipt)
        ids = [item.batch_id for item in receipts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate current cycle batch receipts")
        return tuple(receipts)

    def write(self, receipt: CurrentCycleReceipt) -> Path:
        expected = batch_id(
            receipt.target_execution_date,
            receipt.selected_event_ids,
        )
        if receipt.batch_id != expected:
            raise ValueError("current cycle receipt batch_id does not match event batch")
        path = self._path(receipt.batch_id)
        existing = self.load(receipt.batch_id)
        if existing is not None:
            if existing != receipt:
                raise ValueError("current cycle receipt changed after publication")
            return path
        payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            **asdict(receipt),
            "completed_at": receipt.completed_at.isoformat(),
            "target_execution_date": receipt.target_execution_date.isoformat(),
            "selected_event_ids": list(receipt.selected_event_ids),
            "candidate_ids": list(receipt.candidate_ids),
            "champion_entry_order_ids": list(receipt.champion_entry_order_ids),
            "shadow_strategy_ids": list(receipt.shadow_strategy_ids),
        }
        _atomic_json_write(path, payload)
        return path

    def _path(self, batch_id_value: str) -> Path:
        if not batch_id_value.startswith("batch_"):
            raise ValueError("invalid current cycle batch_id")
        return self.root / f"{batch_id_value}.json"


def verify_receipt_paper_orders(
    receipts: tuple[CurrentCycleReceipt, ...],
    paper_ledger,
) -> ReceiptPaperIntegrityResult:
    """Require every receipt-referenced champion entry order to remain durable.

    A current-cycle receipt is publication authority for queue reconciliation. That
    authority is valid only while every champion entry order named by the receipt is
    still represented by the PAPER ledger, either as a pending order or as a
    completed execution report. This catches ledger deletion/corruption before an
    old receipt can suppress or acknowledge SEC work.
    """

    state = paper_ledger.load()
    pending_ids = {order.order_id for order in state.pending_orders}
    completed_ids = {report.order_id for report in state.completed_reports}
    referenced = 0
    pending_count = 0
    completed_count = 0

    for receipt in receipts:
        missing: list[str] = []
        for order_id in receipt.champion_entry_order_ids:
            referenced += 1
            if order_id in pending_ids:
                pending_count += 1
            elif order_id in completed_ids:
                completed_count += 1
            else:
                missing.append(order_id)
        if missing:
            raise RuntimeError(
                "completed receipt PAPER order integrity failure for "
                f"{receipt.batch_id}: missing champion order IDs {sorted(missing)}"
            )

    return ReceiptPaperIntegrityResult(
        receipt_count=len(receipts),
        referenced_champion_order_count=referenced,
        pending_champion_order_count=pending_count,
        completed_champion_order_count=completed_count,
    )


def reconcile_completed_receipts(
    queue,
    store: FileCurrentCycleReceiptStore,
    *,
    paper_ledger=None,
) -> ReceiptReconciliationResult:
    """Finish queue acknowledgement for batches whose durable receipt already exists.

    Reconciliation intentionally runs before exchange-session classification. This
    prevents a crash after receipt publication but before queue acknowledgement from
    causing an already-evaluated event to be mislabeled as stale after the intended
    market open has passed. When a PAPER ledger is supplied, every published
    champion order is verified before any queue acknowledgement is allowed.
    """

    receipts = store.load_all()
    integrity = (
        verify_receipt_paper_orders(receipts, paper_ledger)
        if paper_ledger is not None
        else ReceiptPaperIntegrityResult(
            receipt_count=len(receipts),
            referenced_champion_order_count=0,
            pending_champion_order_count=0,
            completed_champion_order_count=0,
        )
    )
    pending_ids = {item.event_id for item in queue.pending()}
    matched: list[str] = []
    acknowledged = 0
    for receipt in receipts:
        outstanding = tuple(
            event_id
            for event_id in receipt.selected_event_ids
            if event_id in pending_ids
        )
        if not outstanding:
            continue
        removed = queue.acknowledge(outstanding)
        if removed != len(outstanding):
            raise RuntimeError(
                f"completed receipt acknowledgement was not atomic for {receipt.batch_id}"
            )
        acknowledged += removed
        matched.append(receipt.batch_id)
        pending_ids.difference_update(outstanding)
    return ReceiptReconciliationResult(
        receipt_count=len(receipts),
        matched_receipt_count=len(matched),
        acknowledged_pending_event_count=acknowledged,
        matched_batch_ids=tuple(matched),
        referenced_champion_order_count=integrity.referenced_champion_order_count,
        pending_champion_order_count=integrity.pending_champion_order_count,
        completed_champion_order_count=integrity.completed_champion_order_count,
    )


def batch_id(target_execution_date: date, event_ids: tuple[str, ...]) -> str:
    normalized = tuple(sorted(set(str(item) for item in event_ids if str(item))))
    if not normalized:
        raise ValueError("cannot build current cycle batch_id without event IDs")
    digest = hashlib.sha256(
        (target_execution_date.isoformat() + "|" + "|".join(normalized)).encode("utf-8")
    ).hexdigest()[:24]
    return f"batch_{digest}"


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
