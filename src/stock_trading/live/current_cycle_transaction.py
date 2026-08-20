from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from stock_trading.core import as_utc
from stock_trading.engine import OrderSide
from stock_trading.execution.paper_batch_commit import FilePaperRuntimeBatchCommitStore

from .current_cycle_receipt import (
    CurrentCycleReceipt,
    FileCurrentCycleReceiptStore,
    batch_id,
    verify_receipt_paper_orders,
)


@dataclass(frozen=True, slots=True)
class CurrentCycleTransaction:
    batch_id: str
    prepared_at: datetime
    target_execution_date: date
    selected_event_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    champion_strategy_id: str
    shadow_strategy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prepared_at", as_utc(self.prepared_at))
        if not self.batch_id.strip() or not self.champion_strategy_id.strip():
            raise ValueError("current cycle transaction identity must not be empty")
        expected = batch_id(self.target_execution_date, self.selected_event_ids)
        if self.batch_id != expected:
            raise ValueError("current cycle transaction batch_id does not match event batch")
        if len(self.selected_event_ids) != len(set(self.selected_event_ids)):
            raise ValueError("current cycle transaction contains duplicate selected event IDs")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("current cycle transaction contains duplicate candidate IDs")
        if len(self.shadow_strategy_ids) != len(set(self.shadow_strategy_ids)):
            raise ValueError("current cycle transaction contains duplicate SHADOW strategy IDs")


@dataclass(frozen=True, slots=True)
class TransactionReceiptRecoveryResult:
    transaction_count: int
    recovered_receipt_count: int
    recovered_batch_ids: tuple[str, ...]


class FileCurrentCycleTransactionStore:
    """Durable pre-broker identity for reconstructing a lost batch receipt."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, batch_id_value: str) -> CurrentCycleTransaction | None:
        path = self._path(batch_id_value)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid current cycle transaction: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported current cycle transaction schema")
        try:
            return CurrentCycleTransaction(
                batch_id=str(payload["batch_id"]),
                prepared_at=datetime.fromisoformat(str(payload["prepared_at"])),
                target_execution_date=date.fromisoformat(
                    str(payload["target_execution_date"])
                ),
                selected_event_ids=tuple(str(item) for item in payload["selected_event_ids"]),
                candidate_ids=tuple(str(item) for item in payload["candidate_ids"]),
                champion_strategy_id=str(payload["champion_strategy_id"]),
                shadow_strategy_ids=tuple(
                    str(item) for item in payload.get("shadow_strategy_ids", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid current cycle transaction: {path}") from exc

    def load_all(self) -> tuple[CurrentCycleTransaction, ...]:
        if not self.root.exists():
            return ()
        transactions: list[CurrentCycleTransaction] = []
        for path in sorted(self.root.glob("batch_*.json")):
            transaction = self.load(path.stem)
            if transaction is None:
                raise RuntimeError(
                    f"current cycle transaction disappeared during scan: {path}"
                )
            transactions.append(transaction)
        return tuple(transactions)

    def write(self, transaction: CurrentCycleTransaction) -> Path:
        path = self._path(transaction.batch_id)
        existing = self.load(transaction.batch_id)
        if existing is not None:
            if _transaction_identity(existing) != _transaction_identity(transaction):
                raise ValueError("current cycle transaction changed after preparation")
            return path
        payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            **asdict(transaction),
            "prepared_at": transaction.prepared_at.isoformat(),
            "target_execution_date": transaction.target_execution_date.isoformat(),
            "selected_event_ids": list(transaction.selected_event_ids),
            "candidate_ids": list(transaction.candidate_ids),
            "shadow_strategy_ids": list(transaction.shadow_strategy_ids),
        }
        _atomic_json_write(path, payload)
        return path

    def _path(self, batch_id_value: str) -> Path:
        if not batch_id_value.startswith("batch_"):
            raise ValueError("invalid current cycle batch_id")
        return self.root / f"{batch_id_value}.json"


def recover_submitted_batch_receipts(
    *,
    transaction_store: FileCurrentCycleTransactionStore,
    receipt_store: FileCurrentCycleReceiptStore,
    paper_ledger,
) -> TransactionReceiptRecoveryResult:
    """Publish receipts for batches that durably crossed the PAPER broker boundary.

    Recovery intentionally happens before pending-event session classification. A
    durable batch-tagged BUY proves the broker atomically saved that order. For a
    zero-order decision, the broker writes an explicit runtime batch-commit sidecar
    only after its execute call succeeds. The prepared transaction supplies the
    otherwise non-reversible event/candidate and strategy identities needed to
    reconstruct the exact receipt on a later restart.
    """

    transactions = transaction_store.load_all()
    state = paper_ledger.load()
    commit_store = FilePaperRuntimeBatchCommitStore.for_ledger(paper_ledger)
    submitted_by_id = {order.order_id: order for order in state.submitted_orders}
    recovered: list[str] = []
    for transaction in transactions:
        if receipt_store.load(transaction.batch_id) is not None:
            continue
        broker_commit = commit_store.load(transaction.batch_id)
        if broker_commit is not None:
            missing_committed = sorted(
                order_id
                for order_id in broker_commit.submitted_order_ids
                if order_id not in submitted_by_id
            )
            if missing_committed:
                raise RuntimeError(
                    "PAPER runtime batch commit references missing submitted orders: "
                    f"{missing_committed}"
                )
            wrong_batch = sorted(
                order_id
                for order_id in broker_commit.submitted_order_ids
                if submitted_by_id[order_id].metadata.get("runtime_batch_id")
                != transaction.batch_id
            )
            if wrong_batch:
                raise RuntimeError(
                    "PAPER runtime batch commit order ownership mismatch: "
                    f"{wrong_batch}"
                )
        order_ids = tuple(
            sorted(
                order.order_id
                for order in state.submitted_orders
                if order.side is OrderSide.BUY
                and order.strategy_id == transaction.champion_strategy_id
                and order.metadata.get("runtime_batch_id") == transaction.batch_id
            )
        )
        if not order_ids and broker_commit is None:
            continue
        receipt = CurrentCycleReceipt(
            batch_id=transaction.batch_id,
            completed_at=datetime.now(timezone.utc),
            target_execution_date=transaction.target_execution_date,
            selected_event_ids=transaction.selected_event_ids,
            candidate_ids=transaction.candidate_ids,
            champion_strategy_id=transaction.champion_strategy_id,
            champion_entry_order_ids=order_ids,
            shadow_strategy_ids=transaction.shadow_strategy_ids,
        )
        verify_receipt_paper_orders((receipt,), paper_ledger)
        receipt_store.write(receipt)
        recovered.append(transaction.batch_id)
    return TransactionReceiptRecoveryResult(
        transaction_count=len(transactions),
        recovered_receipt_count=len(recovered),
        recovered_batch_ids=tuple(recovered),
    )


def _transaction_identity(transaction: CurrentCycleTransaction) -> tuple:
    return (
        transaction.batch_id,
        transaction.target_execution_date,
        transaction.selected_event_ids,
        transaction.candidate_ids,
        transaction.champion_strategy_id,
        transaction.shadow_strategy_ids,
    )


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
