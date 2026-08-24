from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

import httpx

from stock_trading.contracts import UsaSpendingClient, UsaSpendingNormalizer
from stock_trading.core import Event, EventType, Source, as_utc
from stock_trading.entities import DuckDbExternalEntityAliases, ExternalEntityAlias
from stock_trading.extraction import FileSemanticCache, QwenSemanticExtractor
from stock_trading.extraction.qwen import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL
from stock_trading.market import CandidateSnapshotBuilder, DuckDbMarketStore, TiingoClient
from stock_trading.market import normalize_company_name
from stock_trading.storage import DuckDbEventStore, FileRawStore

from .candidates import PitCandidateAssembler
from .current_cycle_receipt import batch_id
from .decision_diagnostics import FileStrategyDecisionDiagnosticStore, diagnose_registry
from .forward_evidence_invalidations import (
    FileForwardEvidenceInvalidationStore,
    ForwardEvidenceInvalidation,
)
from .forward_outcomes import refresh_forward_outcome_scorecard
from .run_current_lda_shadow import (
    _atomic_json_write,
    _events_by_ids,
    _jsonable,
    _load_tiingo_token,
    _merge_event_history,
    _modeled_company_ids,
    _modeled_company_name_index,
    _parse_as_of,
    _sync_known_companies,
)
from .run_current_paper_shadow import _load_runtime_config
from .runtime_lock import FileRuntimeLock
from .runtime_state import load_persisted_shadow_registry
from .runtime_strategy_state import FileRuntimeStrategyStateStore
from .session_calendar import XnysExecutionSessionResolver


@dataclass(frozen=True, slots=True)
class UsaSpendingShadowPending:
    event_id: str
    company_id: str
    public_time: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_time", as_utc(self.public_time))
        if not self.event_id.strip() or not self.company_id.strip():
            raise ValueError("USAspending shadow pending identity must not be empty")


@dataclass(frozen=True, slots=True)
class UsaSpendingShadowIntakeState:
    watermark: datetime | None = None
    pending: tuple[UsaSpendingShadowPending, ...] = ()
    handled_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.watermark is not None:
            object.__setattr__(self, "watermark", as_utc(self.watermark))


class FileUsaSpendingShadowIntake:
    """Atomic source watermark + pending + replay tombstones for contract SHADOW intake."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> UsaSpendingShadowIntakeState:
        if not self.path.exists():
            return UsaSpendingShadowIntakeState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid USAspending shadow intake state: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported USAspending shadow intake schema")

        watermark_text = payload.get("watermark")
        watermark = _parse_iso_datetime(watermark_text) if watermark_text else None
        try:
            pending = tuple(
                UsaSpendingShadowPending(
                    event_id=str(item["event_id"]),
                    company_id=str(item["company_id"]),
                    public_time=_parse_iso_datetime(item["public_time"]),
                )
                for item in payload.get("pending", [])
            )
            handled = tuple(str(item) for item in payload.get("handled_event_ids", []))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid USAspending shadow intake payload") from exc

        pending_ids = [item.event_id for item in pending]
        if len(pending_ids) != len(set(pending_ids)):
            raise ValueError("duplicate USAspending shadow pending event IDs")
        if len(handled) != len(set(handled)):
            raise ValueError("duplicate USAspending shadow handled event IDs")
        if set(pending_ids) & set(handled):
            raise ValueError("USAspending shadow event cannot be pending and handled")
        return UsaSpendingShadowIntakeState(
            watermark=watermark,
            pending=tuple(sorted(pending, key=_pending_sort_key)),
            handled_event_ids=tuple(sorted(handled)),
        )

    def commit_poll(self, *, watermark: datetime, events: Iterable[Event]) -> int:
        state = self.load()
        next_watermark = max(as_utc(watermark), state.watermark) if state.watermark else as_utc(watermark)
        pending = {item.event_id: item for item in state.pending}
        handled = set(state.handled_event_ids)
        added = 0
        for event in events:
            if event.source is not Source.USASPENDING or not event.company_id:
                raise ValueError("USAspending shadow intake accepts only mapped USAspending events")
            if event.event_id in handled:
                continue
            item = UsaSpendingShadowPending(
                event_id=event.event_id,
                company_id=event.company_id,
                public_time=event.public_time,
            )
            previous = pending.get(item.event_id)
            if previous is not None and previous != item:
                raise ValueError(f"USAspending pending identity changed for {item.event_id}")
            if previous is None:
                pending[item.event_id] = item
                added += 1
        if next_watermark != state.watermark or added:
            self._save(
                UsaSpendingShadowIntakeState(
                    watermark=next_watermark,
                    pending=tuple(sorted(pending.values(), key=_pending_sort_key)),
                    handled_event_ids=state.handled_event_ids,
                )
            )
        return added

    def pending(self, *, as_of: datetime | None = None) -> tuple[UsaSpendingShadowPending, ...]:
        values = self.load().pending
        if as_of is None:
            return values
        cutoff = as_utc(as_of)
        return tuple(item for item in values if item.public_time <= cutoff)

    def acknowledge(self, event_ids: Iterable[str]) -> int:
        selected = {str(item) for item in event_ids if str(item)}
        if not selected:
            return 0
        state = self.load()
        removed = tuple(item for item in state.pending if item.event_id in selected)
        if not removed:
            return 0
        kept = tuple(item for item in state.pending if item.event_id not in selected)
        handled = tuple(sorted(set(state.handled_event_ids) | {item.event_id for item in removed}))
        self._save(
            UsaSpendingShadowIntakeState(
                watermark=state.watermark,
                pending=kept,
                handled_event_ids=handled,
            )
        )
        return len(removed)

    def dispose_stale(self, event_ids: Iterable[str]) -> int:
        return self.acknowledge(event_ids)

    def _save(self, state: UsaSpendingShadowIntakeState) -> None:
        _atomic_json_write(
            self.path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "watermark": state.watermark.isoformat() if state.watermark else None,
                "pending": [
                    {
                        "event_id": item.event_id,
                        "company_id": item.company_id,
                        "public_time": item.public_time.isoformat(),
                    }
                    for item in sorted(state.pending, key=_pending_sort_key)
                ],
                "handled_event_ids": list(state.handled_event_ids),
            },
        )


@dataclass(frozen=True, slots=True)
class UsaSpendingAwardDiagnostic:
    award_id: str
    generated_award_id: str
    recipient_name: str
    recipient_uei: str
    reason: str


@dataclass(frozen=True, slots=True)
class UsaSpendingTransactionDiagnostic:
    award_id: str
    action_date: str
    modification_number: str
    transaction_amount: str
    reason: str
    match_count: int


@dataclass(frozen=True, slots=True)
class UsaSpendingMappedEventDiagnostic:
    event_id: str
    company_id: str
    modeled_company_name: str
    recipient_name: str
    award_id: str
    transaction_id: str
    action_date: str
    public_time: datetime
    modification_number: str
    obligation_amount: str
    total_obligation: str
    potential_award_amount: str
    agency: str
    subagency: str
    action_type: str
    description: str
    semantic_topics: tuple[str, ...]
    semantic_direction: str
    semantic_importance: float | None
    semantic_company_relevance: float | None
    semantic_confidence: float | None


USASPENDING_DIAGNOSTIC_LIMIT = 10
USASPENDING_MAX_ACTION_LAG_DAYS = 30
USASPENDING_FRESHNESS_MIGRATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class UsaSpendingShadowPollResult:
    award_pages_fetched: int
    awards_seen: int
    candidate_award_count: int
    mapped_award_count: int
    nonmodeled_award_count: int
    unmapped_award_count: int
    unmapped_award_sample: tuple[UsaSpendingAwardDiagnostic, ...]
    transaction_search_pages_fetched: int
    transaction_detail_pages_fetched: int
    transaction_search_row_count: int
    transaction_identity_filtered_row_count: int
    freshness_filtered_transaction_count: int
    matched_transaction_count: int
    unmatched_transaction_count: int
    transaction_diagnostic_sample: tuple[UsaSpendingTransactionDiagnostic, ...]
    mapped_event_count: int
    recent_mapped_event_sample: tuple[UsaSpendingMappedEventDiagnostic, ...]
    semantic_enriched_event_count: int
    recovered_event_count: int
    pending_events_added: int
    pending_event_count: int
    handled_event_count: int
    watermark: datetime | None


@dataclass(frozen=True, slots=True)
class UsaSpendingShadowBatchSelection:
    target_execution_date: date
    selected_event_ids: tuple[str, ...]
    stale_event_ids: tuple[str, ...]
    future_event_ids: tuple[str, ...]


def poll_current_usaspending_shadow(
    *,
    data_root: str | Path,
    experiment_dir: str | Path,
    runtime_dir: str | Path,
    usaspending_client: UsaSpendingClient,
    extractor: QwenSemanticExtractor,
    as_of: datetime,
    initial_lookback_days: int = 1,
    overlap_days: int = 1,
    max_pages: int | None = None,
) -> UsaSpendingShadowPollResult:
    """Discover live contract modifications into isolated SHADOW storage."""

    if initial_lookback_days <= 0:
        raise ValueError("initial_lookback_days must be > 0")
    if overlap_days < 0:
        raise ValueError("overlap_days must be >= 0")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be > 0")

    data_root = Path(data_root)
    experiment_dir = Path(experiment_dir)
    runtime_root = Path(runtime_dir)
    shadow_root = runtime_root / "usaspending_shadow"
    cutoff = as_utc(as_of)
    intake = FileUsaSpendingShadowIntake(shadow_root / "intake.json")
    state = intake.load()
    modified_after = (
        (state.watermark - timedelta(days=overlap_days)).date()
        if state.watermark is not None
        else cutoff.date() - timedelta(days=initial_lookback_days)
    )
    modified_before = cutoff.date()

    modeled_ids = _modeled_company_ids(experiment_dir / "training_rows.jsonl")
    name_index = _modeled_company_name_index(
        data_root / "manifests" / "sec_companies.jsonl",
        modeled_ids,
    )
    shared_aliases_db = data_root / "normalized" / "aliases.duckdb"
    isolated_aliases_db = shadow_root / "aliases.duckdb"
    isolated_alias_store = DuckDbExternalEntityAliases(isolated_aliases_db)
    verified_uei_index = _verified_usaspending_uei_index(
        data_root=data_root,
        runtime_root=runtime_root,
        modeled_ids=modeled_ids,
    )

    raw_store = FileRawStore(shadow_root / "raw")
    event_store = DuckDbEventStore(shadow_root / "events.duckdb")
    invalidated_event_ids = _load_usaspending_freshness_invalidated_event_ids(shadow_root)
    known_events = {
        event.event_id: event
        for event in event_store.all_events()
        if event.event_id not in invalidated_event_ids
    }
    pending_ids = {item.event_id for item in state.pending}
    handled_ids = set(state.handled_event_ids)
    normalizer = UsaSpendingNormalizer()

    award_pages = 0
    awards_seen = 0
    candidate_awards = 0
    mapped_awards = 0
    nonmodeled_awards = 0
    unmapped_awards = 0
    unmapped_sample: list[UsaSpendingAwardDiagnostic] = []
    transaction_search_pages = 0
    transaction_detail_pages = 0
    transaction_search_rows = 0
    transaction_identity_filtered_rows = 0
    freshness_filtered_transactions = 0
    matched_transactions = 0
    unmatched_transactions = 0
    transaction_sample: list[UsaSpendingTransactionDiagnostic] = []
    enriched_events: list[Event] = []
    recovered_events: list[Event] = []

    page = 1
    while True:
        search_raw = usaspending_client.search_contract_awards_page(
            modified_after=modified_after,
            modified_before=modified_before,
            page=page,
            limit=100,
        )
        raw_store.put(search_raw)
        award_pages += 1
        payload = _json_payload(search_raw.content, "USAspending award search")
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("USAspending award search response has no results list")
        awards_seen += len(results)

        for row in results:
            if not isinstance(row, dict):
                continue
            award_search_id = str(row.get("Award ID") or "").strip()
            generated_award_id = str(row.get("generated_internal_id") or "").strip()
            recipient_name = str(row.get("Recipient Name") or "").strip()
            recipient_uei = str(row.get("Recipient UEI") or "").strip().upper()
            if not award_search_id or not generated_award_id:
                unmapped_awards += 1
                _append_award_diagnostic(
                    unmapped_sample,
                    award_search_id,
                    generated_award_id,
                    recipient_name,
                    recipient_uei,
                    "missing_award_identity",
                )
                continue

            direct = _candidate_companies(
                recipient_uei=recipient_uei,
                recipient_name=recipient_name,
                name_index=name_index,
                verified_uei_index=verified_uei_index,
                shared_aliases_db=shared_aliases_db,
                isolated_aliases_db=isolated_aliases_db,
            )
            if not direct and not _possibly_modeled_recipient(recipient_name, name_index):
                continue
            candidate_awards += 1

            award_raw = usaspending_client.fetch_award(generated_award_id)
            raw_store.put(award_raw)
            award = normalizer.parse_award(award_raw)
            resolved_candidates = set(direct)
            for uei, name in (
                (award.recipient_uei, award.recipient_name),
                (award.parent_recipient_uei, award.parent_recipient_name),
            ):
                resolved_candidates.update(
                    _candidate_companies(
                        recipient_uei=uei or "",
                        recipient_name=name or "",
                        name_index=name_index,
                        verified_uei_index=verified_uei_index,
                        shared_aliases_db=shared_aliases_db,
                        isolated_aliases_db=isolated_aliases_db,
                    )
                )

            if len(resolved_candidates) != 1:
                unmapped_awards += 1
                _append_award_diagnostic(
                    unmapped_sample,
                    award_search_id,
                    generated_award_id,
                    award.recipient_name or recipient_name,
                    award.recipient_uei or recipient_uei,
                    "ambiguous_company" if len(resolved_candidates) > 1 else "unresolved_company",
                )
                continue

            company_id = next(iter(resolved_candidates))
            if company_id not in modeled_ids:
                nonmodeled_awards += 1
                continue
            mapped_awards += 1
            _remember_award_ueis(
                isolated_alias_store,
                award=award,
                company_id=company_id,
                name_index=name_index,
            )
            for uei in (award.recipient_uei, award.parent_recipient_uei):
                if uei:
                    verified_uei_index.setdefault(uei.strip().upper(), set()).add(company_id)

            changed_rows: list[dict] = []
            tx_page = 1
            while True:
                tx_search_raw = usaspending_client.search_contract_transactions_page(
                    award_search_id,
                    generated_award_id=generated_award_id,
                    modified_after=modified_after,
                    modified_before=modified_before,
                    page=tx_page,
                    limit=100,
                )
                raw_store.put(tx_search_raw)
                transaction_search_pages += 1
                tx_search_payload = _json_payload(
                    tx_search_raw.content,
                    "USAspending transaction search",
                )
                tx_results = tx_search_payload.get("results")
                if not isinstance(tx_results, list):
                    raise ValueError("USAspending transaction search response has no results list")
                typed_tx_results = [item for item in tx_results if isinstance(item, dict)]
                accepted_rows, filtered_rows = _transaction_search_rows_for_award(
                    typed_tx_results,
                    generated_award_id,
                )
                fresh_rows: list[dict] = []
                for accepted in accepted_rows:
                    freshness_reason = _transaction_action_freshness_reason(
                        accepted,
                        observed_on=cutoff.date(),
                    )
                    if freshness_reason is None:
                        fresh_rows.append(accepted)
                    else:
                        freshness_filtered_transactions += 1
                        _append_transaction_diagnostic(
                            transaction_sample,
                            award_search_id,
                            accepted,
                            freshness_reason,
                            0,
                        )
                changed_rows.extend(fresh_rows)
                transaction_search_rows += len(typed_tx_results)
                transaction_identity_filtered_rows += len(filtered_rows)
                for filtered in filtered_rows:
                    _append_transaction_diagnostic(
                        transaction_sample,
                        award_search_id,
                        filtered,
                        (
                            "different_generated_award"
                            if str(filtered.get("generated_internal_id") or "").strip()
                            else "missing_generated_award_identity"
                        ),
                        0,
                    )
                if not _has_next(tx_search_payload):
                    break
                tx_page += 1

            if not changed_rows:
                continue

            detail_pages: list[tuple[object, list[dict]]] = []
            detail_page = 1
            detail_rows: list[dict] = []
            while True:
                detail_raw = usaspending_client.fetch_transactions(
                    generated_award_id,
                    page=detail_page,
                    limit=5000,
                )
                raw_store.put(detail_raw)
                transaction_detail_pages += 1
                detail_payload = _json_payload(detail_raw.content, "USAspending transactions")
                rows = detail_payload.get("results")
                if not isinstance(rows, list):
                    raise ValueError("USAspending transactions response has no results list")
                typed_rows = [item for item in rows if isinstance(item, dict)]
                detail_pages.append((detail_raw, typed_rows))
                detail_rows.extend(typed_rows)
                if not _has_next(detail_payload):
                    break
                detail_page += 1

            selected_ids: set[str] = set()
            for changed in changed_rows:
                matches = _matching_transaction_ids(changed, detail_rows)
                if len(matches) == 1:
                    selected_ids.add(matches[0])
                    matched_transactions += 1
                else:
                    unmatched_transactions += 1
                    _append_transaction_diagnostic(
                        transaction_sample,
                        award_search_id,
                        changed,
                        "no_detail_match" if not matches else "ambiguous_detail_match",
                        len(matches),
                    )

            if not selected_ids:
                continue

            company_map = {
                uei.strip().upper(): company_id
                for uei in (award.recipient_uei, award.parent_recipient_uei)
                if uei and uei.strip()
            }
            for detail_raw, rows in detail_pages:
                page_ids = {
                    str(item.get("id") or "").strip()
                    for item in rows
                    if str(item.get("id") or "").strip() in selected_ids
                }
                if not page_ids:
                    continue
                for event in normalizer.to_events(
                    detail_raw,
                    award=award,
                    observed_at=cutoff,
                    company_ids_by_uei=company_map,
                    selected_transaction_ids=page_ids,
                ):
                    if event.event_id in handled_ids:
                        continue
                    previous = known_events.get(event.event_id)
                    if previous is not None:
                        if previous.company_id != event.company_id:
                            raise ValueError(
                                f"USAspending event company changed for {event.event_id}"
                            )
                        if previous.semantic is None:
                            raise RuntimeError(
                                f"stored USAspending shadow event lacks semantic data: {event.event_id}"
                            )
                        if event.event_id not in pending_ids:
                            recovered_events.append(previous)
                        continue
                    semantic = extractor.extract(
                        normalizer.semantic_text(event),
                        context="US federal government contract modification",
                    )
                    enriched = event.model_copy(update={"semantic": semantic})
                    enriched_events.append(enriched)
                    known_events[enriched.event_id] = enriched

        if not _has_next(payload):
            break
        if max_pages is not None and page >= max_pages:
            raise RuntimeError(
                "USAspending shadow max_pages reached before award pagination completed; "
                "intake watermark was not advanced"
            )
        page += 1

    # Persist enriched immutable events before the atomic source state. If a crash
    # occurs between these operations, the overlap poll recovers the stored event
    # without another Qwen call and reconstitutes pending state.
    event_store.put_many(enriched_events)
    commit_events = tuple(enriched_events) + tuple(recovered_events)
    pending_added = intake.commit_poll(watermark=cutoff, events=commit_events)
    final_state = intake.load()
    recent_mapped_event_sample = _recent_mapped_event_sample(
        known_events.values(),
        name_index=name_index,
    )
    return UsaSpendingShadowPollResult(
        award_pages_fetched=award_pages,
        awards_seen=awards_seen,
        candidate_award_count=candidate_awards,
        mapped_award_count=mapped_awards,
        nonmodeled_award_count=nonmodeled_awards,
        unmapped_award_count=unmapped_awards,
        unmapped_award_sample=tuple(unmapped_sample),
        transaction_search_pages_fetched=transaction_search_pages,
        transaction_detail_pages_fetched=transaction_detail_pages,
        transaction_search_row_count=transaction_search_rows,
        transaction_identity_filtered_row_count=transaction_identity_filtered_rows,
        freshness_filtered_transaction_count=freshness_filtered_transactions,
        matched_transaction_count=matched_transactions,
        unmatched_transaction_count=unmatched_transactions,
        transaction_diagnostic_sample=tuple(transaction_sample),
        mapped_event_count=len(enriched_events) + len(recovered_events),
        recent_mapped_event_sample=recent_mapped_event_sample,
        semantic_enriched_event_count=len(enriched_events),
        recovered_event_count=len(recovered_events),
        pending_events_added=pending_added,
        pending_event_count=len(final_state.pending),
        handled_event_count=len(final_state.handled_event_ids),
        watermark=final_state.watermark,
    )


def select_usaspending_shadow_batch(
    intake: FileUsaSpendingShadowIntake,
    *,
    resolver: XnysExecutionSessionResolver,
    as_of: datetime,
) -> UsaSpendingShadowBatchSelection:
    cutoff = as_utc(as_of)
    target = resolver.cycle_execution_date(cutoff)
    selected: list[str] = []
    stale: list[str] = []
    future: list[str] = []
    for item in intake.pending(as_of=cutoff):
        intended = resolver.execution_date(item.public_time)
        if intended < target:
            stale.append(item.event_id)
        elif intended > target:
            future.append(item.event_id)
        else:
            selected.append(item.event_id)
    return UsaSpendingShadowBatchSelection(
        target_execution_date=target,
        selected_event_ids=tuple(selected),
        stale_event_ids=tuple(stale),
        future_event_ids=tuple(future),
    )


def run_current_usaspending_shadow(
    *,
    data_root: str | Path = "data",
    experiment_dir: str | Path = "data/experiments/lightgbm_holdout_250_v2",
    runtime_dir: str | Path = "data/runtime",
    as_of: datetime | None = None,
    initial_lookback_days: int = 1,
    overlap_days: int = 1,
    max_pages: int | None = None,
    qwen_base_url: str | None = None,
    qwen_model: str | None = None,
) -> dict:
    """Collect and score live USAspending contract changes under the shared lock."""

    runtime_root = Path(runtime_dir)
    lock = FileRuntimeLock(runtime_root / "current_pipeline.lock")
    if not lock.acquire():
        return {
            "status": "runtime_busy",
            "authority": "shadow_only_no_paper",
            "evidence_source": "usaspending_shadow",
            "runtime_lock": {"acquired": False, "holder": lock.holder()},
        }
    try:
        result = _run_current_usaspending_shadow_locked(
            data_root=data_root,
            experiment_dir=experiment_dir,
            runtime_dir=runtime_root,
            as_of=as_of,
            initial_lookback_days=initial_lookback_days,
            overlap_days=overlap_days,
            max_pages=max_pages,
            qwen_base_url=qwen_base_url,
            qwen_model=qwen_model,
        )
        return {**result, "runtime_lock": {"acquired": True}}
    finally:
        lock.release()


def _run_current_usaspending_shadow_locked(
    *,
    data_root: str | Path,
    experiment_dir: str | Path,
    runtime_dir: str | Path,
    as_of: datetime | None,
    initial_lookback_days: int,
    overlap_days: int,
    max_pages: int | None,
    qwen_base_url: str | None,
    qwen_model: str | None,
) -> dict:
    data_root = Path(data_root)
    experiment_dir = Path(experiment_dir)
    runtime_root = Path(runtime_dir)
    cutoff = as_utc(as_of or datetime.now(timezone.utc))
    shadow_root = runtime_root / "usaspending_shadow"
    resolved_qwen_url = (
        qwen_base_url or os.environ.get("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL)
    ).strip()
    resolved_qwen_model = (
        qwen_model or os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL)
    ).strip()

    resolver = XnysExecutionSessionResolver()
    config = _load_runtime_config(runtime_root)
    market_store = DuckDbMarketStore(Path(str(config["market_db"])))
    market_store.enable_read_cache(max_series=32)
    intake = FileUsaSpendingShadowIntake(shadow_root / "intake.json")
    freshness_reconciliation = _reconcile_usaspending_action_freshness_v1(
        runtime_root,
        cutoff=cutoff,
    )

    try:
        with UsaSpendingClient() as client, QwenSemanticExtractor(
            cache=FileSemanticCache(data_root / "cache" / "semantic"),
            base_url=resolved_qwen_url,
            model=resolved_qwen_model,
            extractor_version="semantic-v1",
        ) as extractor:
            poll = poll_current_usaspending_shadow(
                data_root=data_root,
                experiment_dir=experiment_dir,
                runtime_dir=runtime_root,
                usaspending_client=client,
                extractor=extractor,
                as_of=cutoff,
                initial_lookback_days=initial_lookback_days,
                overlap_days=overlap_days,
                max_pages=max_pages,
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "USAspending shadow enrichment requires reachable USAspending/Qwen HTTP endpoints"
        ) from exc

    selection = select_usaspending_shadow_batch(intake, resolver=resolver, as_of=cutoff)
    stale_removed = intake.dispose_stale(selection.stale_event_ids)
    base_payload = {
        "as_of": cutoff.isoformat(),
        "authority": "shadow_only_no_paper",
        "evidence_source": "usaspending_shadow",
        "qwen_model": resolved_qwen_model,
        "freshness_policy": {
            "max_action_lag_days": USASPENDING_MAX_ACTION_LAG_DAYS,
            "discovery_date_type": "last_modified_date",
            "economic_date_type": "action_date",
        },
        "freshness_reconciliation": freshness_reconciliation,
        "poll": _jsonable(asdict(poll)),
        "selection": {
            "target_execution_date": selection.target_execution_date.isoformat(),
            "selected_event_count": len(selection.selected_event_ids),
            "selected_event_ids": list(selection.selected_event_ids),
            "stale_event_count": len(selection.stale_event_ids),
            "stale_event_ids": list(selection.stale_event_ids),
            "stale_events_disposed": stale_removed,
            "future_event_count": len(selection.future_event_ids),
            "future_event_ids": list(selection.future_event_ids),
        },
    }
    if not selection.selected_event_ids:
        scorecard = refresh_forward_outcome_scorecard(
            data_root=data_root,
            runtime_dir=runtime_root,
            as_of=cutoff,
        )
        return {
            **base_payload,
            "status": "no_actionable_usaspending_shadow_batch",
            "scored_event_count": 0,
            "remaining_pending_count": len(intake.pending()),
            "forward_outcomes": scorecard,
        }

    shadow_event_store = DuckDbEventStore(shadow_root / "events.duckdb")
    selected_events = _events_by_ids(shadow_event_store, selection.selected_event_ids)
    selected_company_ids = tuple(sorted({item.company_id for item in selected_events if item.company_id}))
    completed_session = resolver.last_completed_session(cutoff)
    shadow_raw_store = FileRawStore(shadow_root / "raw")
    with TiingoClient(_load_tiingo_token(data_root)) as tiingo_client:
        market_sync = _sync_known_companies(
            selected_company_ids,
            target_execution_date=selection.target_execution_date,
            completed_session=completed_session,
            benchmark_security_id=str(config["benchmark_security_id"]),
            market_store=market_store,
            tiingo_client=tiingo_client,
            raw_store=shadow_raw_store,
        )
    market_store.clear_read_cache()
    ready_set = set(market_sync["ready_company_ids"])
    ready_events = tuple(event for event in selected_events if event.company_id in ready_set)
    unready_event_ids = tuple(
        event.event_id for event in selected_events if event.company_id not in ready_set
    )
    if not ready_events:
        return {
            **base_payload,
            "status": "no_market_ready_usaspending_shadow_batch",
            "market_sync": market_sync,
            "scored_event_count": 0,
            "unready_event_ids": list(unready_event_ids),
            "remaining_pending_count": len(intake.pending()),
        }

    current_batch_id = batch_id(
        selection.target_execution_date,
        tuple(sorted(event.event_id for event in ready_events)),
    )
    diagnostic_root = runtime_root / "decision_diagnostics"
    diagnostic_path = diagnostic_root / f"{current_batch_id}.json"
    if diagnostic_path.exists():
        acknowledged = intake.acknowledge(event.event_id for event in ready_events)
        scorecard = refresh_forward_outcome_scorecard(
            data_root=data_root,
            runtime_dir=runtime_root,
            as_of=cutoff,
        )
        return {
            **base_payload,
            "status": "completed_from_existing_usaspending_shadow_diagnostic",
            "batch_id": current_batch_id,
            "diagnostic_path": str(diagnostic_path),
            "market_sync": market_sync,
            "scored_event_count": len(ready_events),
            "unready_event_ids": list(unready_event_ids),
            "acknowledged_scored_event_count": acknowledged,
            "remaining_pending_count": len(intake.pending()),
            "forward_outcomes": scorecard,
        }

    authoritative_events = DuckDbEventStore(
        data_root / "normalized" / "events.duckdb"
    ).all_events(company_ids=list(ready_set))
    histories: list[Iterable[Event]] = [authoritative_events]
    invalidated_usaspending_event_ids = _load_usaspending_freshness_invalidated_event_ids(
        shadow_root
    )
    for path in (
        runtime_root / "lda_shadow" / "events.duckdb",
        shadow_root / "events.duckdb",
    ):
        if path.exists():
            history_events = DuckDbEventStore(path).all_events(company_ids=list(ready_set))
            if path == shadow_root / "events.duckdb":
                history_events = tuple(
                    event
                    for event in history_events
                    if event.event_id not in invalidated_usaspending_event_ids
                )
            histories.append(history_events)
    all_events = _merge_event_history(*histories)
    assembly = PitCandidateAssembler(
        CandidateSnapshotBuilder(
            market_store,
            benchmark_security_id=str(config["benchmark_security_id"]),
        )
    ).assemble(
        ready_events,
        all_events=all_events,
        as_of=cutoff,
        execution_date=selection.target_execution_date,
    )
    if assembly.candidate_count != assembly.affected_company_count:
        raise RuntimeError(
            "USAspending shadow candidate assembly is incomplete; selected events remain pending"
        )

    loaded = load_persisted_shadow_registry(runtime_dir=runtime_root)
    restored_strategy_ids = FileRuntimeStrategyStateStore(
        runtime_root / "strategy_state"
    ).restore_registry(loaded.registry)
    diagnostics = diagnose_registry(loaded.registry, assembly.candidates)
    written_path = FileStrategyDecisionDiagnosticStore(diagnostic_root).write(
        batch_id=current_batch_id,
        as_of=cutoff,
        target_execution_date=selection.target_execution_date,
        diagnostics=diagnostics,
        evidence_source="usaspending_shadow",
    )
    acknowledged = intake.acknowledge(event.event_id for event in ready_events)
    scorecard = refresh_forward_outcome_scorecard(
        data_root=data_root,
        runtime_dir=runtime_root,
        as_of=cutoff,
    )
    return {
        **base_payload,
        "status": "completed",
        "batch_id": current_batch_id,
        "diagnostic_path": str(written_path),
        "market_sync": market_sync,
        "candidate_assembly": {
            "trigger_event_count": assembly.trigger_event_count,
            "affected_company_count": assembly.affected_company_count,
            "candidate_count": assembly.candidate_count,
            "candidate_ids": [item.candidate_id for item in assembly.candidates],
        },
        "restored_strategy_ids": list(restored_strategy_ids),
        "strategies": [
            {
                "strategy_id": item.strategy_id,
                "candidate_count": item.candidate_count,
                "emitted_opportunity_count": item.emitted_opportunity_count,
            }
            for item in diagnostics
        ],
        "scored_event_count": len(ready_events),
        "unready_event_ids": list(unready_event_ids),
        "acknowledged_scored_event_count": acknowledged,
        "remaining_pending_count": len(intake.pending()),
        "forward_outcomes": scorecard,
    }


def _verified_usaspending_uei_index(
    *,
    data_root: Path,
    runtime_root: Path,
    modeled_ids: set[str],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    paths = [data_root / "normalized" / "events.duckdb"]
    paths.extend(
        path
        for path in (
            runtime_root / "usaspending_shadow" / "events.duckdb",
        )
        if path.exists()
    )
    for path in paths:
        if not path.exists():
            continue
        for event in DuckDbEventStore(path).all_events(
            event_types=(EventType.GOVERNMENT_CONTRACT,)
        ):
            if event.source is not Source.USASPENDING or not event.company_id:
                continue
            if event.company_id not in modeled_ids:
                continue
            uei = str(event.payload.recipient_uei or "").strip().upper()
            if uei:
                result.setdefault(uei, set()).add(event.company_id)
    return result


def _candidate_companies(
    *,
    recipient_uei: str,
    recipient_name: str,
    name_index: dict[str, tuple[str, ...]],
    verified_uei_index: dict[str, set[str]],
    shared_aliases_db: Path,
    isolated_aliases_db: Path,
) -> set[str]:
    result: set[str] = set()
    normalized_uei = recipient_uei.strip().upper()
    if normalized_uei:
        for path in (isolated_aliases_db, shared_aliases_db):
            alias = _read_usaspending_alias(path, normalized_uei)
            if alias:
                result.add(alias)
        result.update(verified_uei_index.get(normalized_uei, set()))
    normalized_name = normalize_company_name(recipient_name)
    if normalized_name:
        matches = name_index.get(normalized_name, ())
        if len(matches) == 1:
            result.add(matches[0])
    return result


def _read_usaspending_alias(path: Path, external_id: str) -> str | None:
    if not path.exists():
        return None
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for USAspending alias lookup") from exc
    with duckdb.connect(str(path), read_only=True) as connection:
        table = connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'external_entity_aliases'
            """
        ).fetchone()
        if table is None or int(table[0]) == 0:
            return None
        row = connection.execute(
            """
            SELECT company_id FROM external_entity_aliases
            WHERE source = ? AND external_id = ?
            """,
            [Source.USASPENDING.value, external_id.strip().upper()],
        ).fetchone()
    return str(row[0]) if row is not None else None


GENERIC_RECIPIENT_TOKENS = frozenset(
    {
        "AMERICAN",
        "COMPANY",
        "CORPORATION",
        "FEDERAL",
        "GENERAL",
        "GLOBAL",
        "GROUP",
        "HOLDING",
        "HOLDINGS",
        "INDUSTRIES",
        "INDUSTRY",
        "INTERNATIONAL",
        "LIMITED",
        "NATIONAL",
        "SERVICES",
        "SERVICE",
        "SOLUTIONS",
        "SYSTEMS",
        "TECHNOLOGIES",
        "TECHNOLOGY",
        "UNITED",
    }
)


def _possibly_modeled_recipient(
    recipient_name: str,
    name_index: dict[str, tuple[str, ...]],
) -> bool:
    """Conservative detail-fetch prefilter; never grants company identity."""

    candidate = normalize_company_name(recipient_name)
    if not candidate:
        return False
    candidate_tokens = _distinctive_recipient_tokens(candidate)
    if not candidate_tokens:
        return False
    for modeled_name in name_index:
        if candidate == modeled_name:
            return True
        if candidate.startswith(modeled_name) or modeled_name.startswith(candidate):
            return True
        modeled_tokens = _distinctive_recipient_tokens(modeled_name)
        if not modeled_tokens:
            continue
        # A subsidiary such as "Microsoft Federal" can justify a detail fetch
        # for modeled "Microsoft", but generic roots such as "National" cannot.
        if candidate_tokens[0] == modeled_tokens[0] and len(candidate_tokens[0]) >= 6:
            return True
    return False


def _distinctive_recipient_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in value.split()
        if len(token) >= 4 and token not in GENERIC_RECIPIENT_TOKENS
    )


def _remember_award_ueis(
    alias_store: DuckDbExternalEntityAliases,
    *,
    award,
    company_id: str,
    name_index: dict[str, tuple[str, ...]],
) -> None:
    exact_parent = _unique_name_company(award.parent_recipient_name, name_index)
    exact_recipient = _unique_name_company(award.recipient_name, name_index)
    parent_confirms = exact_parent == company_id
    recipient_confirms = exact_recipient == company_id
    if not parent_confirms and not recipient_confirms:
        return
    basis = (
        "USAspending exact modeled recipient-name match"
        if recipient_confirms
        else "USAspending explicit parent exact modeled issuer-name match"
    )
    for uei, display_name in (
        (award.recipient_uei, award.recipient_name),
        (award.parent_recipient_uei, award.parent_recipient_name),
    ):
        if not uei or not uei.strip():
            continue
        alias_store.add(
            ExternalEntityAlias(
                source=Source.USASPENDING,
                external_id=uei,
                company_id=company_id,
                display_name=display_name,
                resolution_basis=basis,
            )
        )


def _unique_name_company(
    value: str | None,
    name_index: dict[str, tuple[str, ...]],
) -> str | None:
    if not value:
        return None
    matches = name_index.get(normalize_company_name(value), ())
    return matches[0] if len(matches) == 1 else None


def _recent_mapped_event_sample(
    events: Iterable[Event],
    *,
    name_index: dict[str, tuple[str, ...]],
    limit: int = USASPENDING_DIAGNOSTIC_LIMIT,
) -> tuple[UsaSpendingMappedEventDiagnostic, ...]:
    if limit <= 0:
        return ()
    eligible = [
        event
        for event in events
        if event.source is Source.USASPENDING
        and event.event_type is EventType.GOVERNMENT_CONTRACT
        and event.company_id
        and event.semantic is not None
    ]
    eligible.sort(
        key=lambda event: (event.public_time, event.event_time, event.event_id),
        reverse=True,
    )
    diagnostics: list[UsaSpendingMappedEventDiagnostic] = []
    for event in eligible[:limit]:
        payload = event.payload
        semantic = event.semantic
        assert semantic is not None
        diagnostics.append(
            UsaSpendingMappedEventDiagnostic(
                event_id=event.event_id,
                company_id=event.company_id or '',
                modeled_company_name=_modeled_company_display_name(
                    event.company_id or '',
                    name_index,
                ),
                recipient_name=str(payload.recipient_name or ''),
                award_id=str(payload.award_id or ''),
                transaction_id=str(payload.transaction_id or ''),
                action_date=event.event_time.date().isoformat(),
                public_time=event.public_time,
                modification_number=str(payload.modification_number or ''),
                obligation_amount=_diagnostic_decimal(payload.obligation_amount),
                total_obligation=_diagnostic_decimal(payload.total_obligation),
                potential_award_amount=_diagnostic_decimal(payload.potential_award_amount),
                agency=str(payload.agency or ''),
                subagency=str(payload.subagency or ''),
                action_type=str(payload.action_type or ''),
                description=_diagnostic_description(payload.description),
                semantic_topics=tuple(semantic.topics),
                semantic_direction=semantic.direction.value,
                semantic_importance=float(semantic.importance),
                semantic_company_relevance=float(semantic.company_relevance),
                semantic_confidence=float(semantic.confidence),
            )
        )
    return tuple(diagnostics)


def _modeled_company_display_name(
    company_id: str,
    name_index: dict[str, tuple[str, ...]],
) -> str:
    names = [name for name, matches in name_index.items() if company_id in matches]
    if not names:
        return company_id
    return max(names, key=lambda name: (len(name), name))


def _diagnostic_decimal(value) -> str:
    return '' if value is None else str(value)


def _diagnostic_description(value, *, limit: int = 240) -> str:
    text = str(value or '').strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + '...'


def _transaction_action_freshness_reason(
    row: dict,
    *,
    observed_on: date,
    max_action_lag_days: int = USASPENDING_MAX_ACTION_LAG_DAYS,
) -> str | None:
    if max_action_lag_days < 0:
        raise ValueError("max_action_lag_days must be >= 0")
    text = str(row.get("Action Date") or "").strip()[:10]
    try:
        action_date = date.fromisoformat(text)
    except ValueError:
        return "invalid_action_date"
    lag_days = (observed_on - action_date).days
    if lag_days < 0:
        return "future_action_date"
    if lag_days > max_action_lag_days:
        return "stale_action_date"
    return None


def _load_usaspending_freshness_migration(shadow_root: Path) -> dict | None:
    path = shadow_root / "action_freshness_v1.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid USAspending action freshness migration: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != USASPENDING_FRESHNESS_MIGRATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported USAspending action freshness migration schema")
    event_ids = payload.get("invalidated_event_ids")
    batch_ids = payload.get("invalidated_forward_batch_ids")
    if not isinstance(event_ids, list) or not isinstance(batch_ids, list):
        raise ValueError("invalid USAspending action freshness migration payload")
    return payload


def _load_usaspending_freshness_invalidated_event_ids(shadow_root: Path) -> frozenset[str]:
    payload = _load_usaspending_freshness_migration(shadow_root)
    if payload is None:
        return frozenset()
    return frozenset(str(item) for item in payload["invalidated_event_ids"])


def _reconcile_usaspending_action_freshness_v1(
    runtime_root: Path,
    *,
    cutoff: datetime,
) -> dict:
    shadow_root = runtime_root / "usaspending_shadow"
    existing = _load_usaspending_freshness_migration(shadow_root)
    if existing is not None:
        return {
            "applied_now": False,
            "invalidated_stored_event_count": len(existing["invalidated_event_ids"]),
            "invalidated_forward_batch_count": len(existing["invalidated_forward_batch_ids"]),
            "invalidated_event_ids": list(existing["invalidated_event_ids"]),
            "invalidated_forward_batch_ids": list(existing["invalidated_forward_batch_ids"]),
        }

    event_path = shadow_root / "events.duckdb"
    stale_events: list[Event] = []
    if event_path.exists():
        for event in DuckDbEventStore(event_path).all_events():
            if event.source is not Source.USASPENDING:
                continue
            lag_days = (event.public_time.date() - event.event_time.date()).days
            if lag_days > USASPENDING_MAX_ACTION_LAG_DAYS:
                stale_events.append(event)
    invalidated_event_ids = tuple(sorted(event.event_id for event in stale_events))

    invalidated_forward_batch_ids: tuple[str, ...] = ()
    if invalidated_event_ids:
        diagnostic_root = runtime_root / "decision_diagnostics"
        invalidations: list[ForwardEvidenceInvalidation] = []
        for path in sorted(diagnostic_root.glob("batch_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid forward decision diagnostic: {path}") from exc
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError(f"unsupported forward decision diagnostic: {path}")
            if str(payload.get("evidence_source") or "").strip() != "usaspending_shadow":
                continue
            batch = str(payload.get("batch_id") or "")
            if batch != path.stem:
                raise ValueError(f"forward diagnostic batch identity mismatch: {path}")
            invalidations.append(
                ForwardEvidenceInvalidation(
                    batch_id=batch,
                    evidence_source="usaspending_shadow",
                    reason="usaspending_pre_action_freshness_guard_v1",
                    invalidated_at=cutoff,
                )
            )
        FileForwardEvidenceInvalidationStore(
            runtime_root / "forward_evidence_invalidations.json"
        ).add_many(tuple(invalidations))
        invalidated_forward_batch_ids = tuple(sorted(item.batch_id for item in invalidations))
        FileUsaSpendingShadowIntake(shadow_root / "intake.json").dispose_stale(
            invalidated_event_ids
        )

    migration_payload = {
        "schema_version": USASPENDING_FRESHNESS_MIGRATION_SCHEMA_VERSION,
        "migrated_at": cutoff.isoformat(),
        "max_action_lag_days": USASPENDING_MAX_ACTION_LAG_DAYS,
        "invalidated_event_ids": list(invalidated_event_ids),
        "invalidated_forward_batch_ids": list(invalidated_forward_batch_ids),
    }
    _atomic_json_write(shadow_root / "action_freshness_v1.json", migration_payload)
    return {
        "applied_now": True,
        "invalidated_stored_event_count": len(invalidated_event_ids),
        "invalidated_forward_batch_count": len(invalidated_forward_batch_ids),
        "invalidated_event_ids": list(invalidated_event_ids),
        "invalidated_forward_batch_ids": list(invalidated_forward_batch_ids),
    }


def _transaction_search_rows_for_award(
    rows: list[dict],
    generated_award_id: str,
) -> tuple[list[dict], list[dict]]:
    """Partition transaction-search rows by exact USAspending award identity.

    Display Award IDs/PIIDs can be reused beneath different parent vehicles.
    The generated internal award ID is therefore the authority for deciding
    whether a search row belongs to the award currently being normalized.
    """

    expected = generated_award_id.strip()
    if not expected:
        raise ValueError("generated_award_id must not be empty")
    accepted: list[dict] = []
    filtered: list[dict] = []
    for row in rows:
        observed = str(row.get("generated_internal_id") or "").strip()
        if observed == expected:
            accepted.append(row)
        else:
            filtered.append(row)
    return accepted, filtered


def _matching_transaction_ids(search_row: dict, detail_rows: list[dict]) -> list[str]:
    action_date = str(search_row.get("Action Date") or "").strip()[:10]
    mod = _text_key(search_row.get("Mod"))
    amount = _decimal_key(search_row.get("Transaction Amount"))
    description = _normalized_text(search_row.get("Transaction Description"))
    action_type = _normalized_text(search_row.get("Action Type"))

    matches: list[str] = []
    for row in detail_rows:
        transaction_id = str(row.get("id") or "").strip()
        if not transaction_id:
            continue
        if action_date and str(row.get("action_date") or "").strip()[:10] != action_date:
            continue
        if mod and _text_key(row.get("modification_number")) != mod:
            continue
        if amount is not None and _decimal_key(row.get("federal_action_obligation")) != amount:
            continue
        if description:
            detail_description = _normalized_text(row.get("description"))
            if detail_description and detail_description != description:
                continue
        if action_type:
            detail_types = {
                _normalized_text(row.get("action_type")),
                _normalized_text(row.get("action_type_description")),
            }
            if any(detail_types) and action_type not in detail_types:
                continue
        matches.append(transaction_id)
    return sorted(set(matches))


def _append_award_diagnostic(
    target: list[UsaSpendingAwardDiagnostic],
    award_id: str,
    generated_award_id: str,
    recipient_name: str,
    recipient_uei: str,
    reason: str,
) -> None:
    if len(target) >= USASPENDING_DIAGNOSTIC_LIMIT:
        return
    target.append(
        UsaSpendingAwardDiagnostic(
            award_id=award_id,
            generated_award_id=generated_award_id,
            recipient_name=recipient_name,
            recipient_uei=recipient_uei,
            reason=reason,
        )
    )


def _append_transaction_diagnostic(
    target: list[UsaSpendingTransactionDiagnostic],
    award_id: str,
    row: dict,
    reason: str,
    match_count: int,
) -> None:
    if len(target) >= USASPENDING_DIAGNOSTIC_LIMIT:
        return
    target.append(
        UsaSpendingTransactionDiagnostic(
            award_id=award_id,
            action_date=str(row.get("Action Date") or ""),
            modification_number=str(row.get("Mod") or ""),
            transaction_amount=str(row.get("Transaction Amount") or ""),
            reason=reason,
            match_count=match_count,
        )
    )


def _json_payload(content: bytes | str, label: str) -> dict:
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} response must be an object")
    return payload


def _has_next(payload: dict) -> bool:
    metadata = payload.get("page_metadata")
    return bool(isinstance(metadata, dict) and metadata.get("hasNext"))


def _parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("USAspending shadow timestamps must include timezone")
    return as_utc(parsed)


def _text_key(value) -> str:
    return str(value or "").strip().upper()


def _decimal_key(value) -> Decimal | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _normalized_text(value) -> str:
    return " ".join(re.findall(r"[A-Z0-9]+", str(value or "").upper()))


def _pending_sort_key(item: UsaSpendingShadowPending):
    return item.public_time, item.event_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poll live USAspending contract modifications into isolated SHADOW storage, "
            "score them read-only with the current strategy registry, and publish forward evidence."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("data/experiments/lightgbm_holdout_250_v2"),
    )
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--initial-lookback-days", type=int, default=1)
    parser.add_argument("--overlap-days", type=int, default=1)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--as-of")
    parser.add_argument(
        "--qwen-base-url",
        default=os.environ.get("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL),
    )
    args = parser.parse_args()
    result = run_current_usaspending_shadow(
        data_root=args.data_root,
        experiment_dir=args.experiment_dir,
        runtime_dir=args.runtime_dir,
        as_of=_parse_as_of(args.as_of),
        initial_lookback_days=args.initial_lookback_days,
        overlap_days=args.overlap_days,
        max_pages=args.max_pages,
        qwen_base_url=args.qwen_base_url,
        qwen_model=args.qwen_model,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
