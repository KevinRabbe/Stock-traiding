from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Protocol

from stock_trading.core import Event, EventType, as_utc
from stock_trading.engine import FeatureSnapshot
from stock_trading.features import (
    CompanyEventIndex,
    build_alternative_features,
    build_congress_features,
    build_insider_features,
)
from stock_trading.market import CandidateSnapshot, CandidateSnapshotBuilder
from stock_trading.market.execution_time import decision_market_date
from stock_trading.ml.dataset import (
    build_opportunity_trigger_features,
    build_research_interactions,
)
from stock_trading.ml.system_context import augment_system_context_features
from stock_trading.storage import DuckDbEventStore


_TRIGGER_TYPES = (
    EventType.INSIDER_TRANSACTION,
    EventType.GOVERNMENT_CONTRACT,
    EventType.LOBBYING_ACTIVITY,
)


@dataclass(frozen=True, slots=True)
class PitFeatureRow:
    """Label-free opportunity row used by current PAPER/SHADOW candidate assembly.

    Its structural fields intentionally match the subset consumed by the existing
    opportunity-history/system-context augmenters. No realized return, exit date,
    alpha, downside, or other outcome can exist on this object.
    """

    event_id: str
    company_id: str
    security_id: str
    decision_time: datetime
    execution_date: date
    features: dict[str, float | None]
    trigger_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PitCandidateAssembly:
    as_of: datetime
    execution_date: date
    trigger_event_count: int
    affected_company_count: int
    context_opportunity_count: int
    candidate_count: int
    candidates: tuple[FeatureSnapshot, ...]


class ExecutionSessionResolver(Protocol):
    """Resolve the next executable U.S. equity session without using future prices."""

    def execution_date(self, as_of: datetime) -> date: ...


class TriggerEventBatchProvider(Protocol):
    """Return newly actionable normalized trigger events known by ``as_of``."""

    def events(self, as_of: datetime) -> tuple[Event, ...]: ...


@dataclass(frozen=True, slots=True)
class FixedExecutionSessionResolver:
    """Explicit session resolver for deterministic tests/manual PAPER intake.

    Production scheduling can later replace this object with an exchange-calendar
    adapter without changing candidate feature construction.
    """

    session_date: date

    def execution_date(self, as_of: datetime) -> date:
        del as_of
        return self.session_date


@dataclass(frozen=True, slots=True)
class StaticTriggerEventBatchProvider:
    trigger_events: tuple[Event, ...]

    def events(self, as_of: datetime) -> tuple[Event, ...]:
        cutoff = as_utc(as_of)
        return tuple(
            event for event in self.trigger_events if as_utc(event.public_time) <= cutoff
        )


@dataclass(frozen=True, slots=True)
class _OpportunitySlice:
    snapshot: CandidateSnapshot
    triggers: tuple[Event, ...]


class PitCandidateAssembler:
    """Build current candidates with training-equivalent PIT feature semantics.

    The assembler reconstructs the prior opportunity sequence for affected
    companies before applying opportunity-history and same-session system context.
    Historical opportunities use actual stored next-session bars only to resolve
    their already-past execution dates. The target current session is supplied
    explicitly and its price/bar is never read or required.
    """

    def __init__(
        self,
        snapshot_builder: CandidateSnapshotBuilder,
        *,
        enable_congress_features: bool = False,
    ) -> None:
        self.snapshot_builder = snapshot_builder
        self.enable_congress_features = enable_congress_features

    def assemble(
        self,
        trigger_events: tuple[Event, ...],
        *,
        all_events: tuple[Event, ...],
        as_of: datetime,
        execution_date: date,
    ) -> PitCandidateAssembly:
        cutoff = as_utc(as_of)
        triggers = tuple(
            sorted(
                (
                    event
                    for event in trigger_events
                    if event.event_type in _TRIGGER_TYPES
                    and event.company_id
                    and as_utc(event.public_time) <= cutoff
                ),
                key=lambda event: (event.public_time, event.event_id),
            )
        )
        affected_companies = tuple(
            sorted({event.company_id for event in triggers if event.company_id})
        )
        if not affected_companies:
            return PitCandidateAssembly(
                as_of=cutoff,
                execution_date=execution_date,
                trigger_event_count=0,
                affected_company_count=0,
                context_opportunity_count=0,
                candidate_count=0,
                candidates=(),
            )

        company_set = set(affected_companies)
        history = tuple(
            sorted(
                (
                    event
                    for event in all_events
                    if event.company_id in company_set
                    and as_utc(event.public_time) <= cutoff
                ),
                key=lambda event: (event.public_time, event.event_id),
            )
        )
        history_ids = {event.event_id for event in history}
        missing_trigger_ids = sorted(
            event.event_id for event in triggers if event.event_id not in history_ids
        )
        if missing_trigger_ids:
            raise ValueError(
                "current trigger batch is missing from supplied PIT event history: "
                f"{missing_trigger_ids[:5]}"
            )

        rows = self._base_rows(history, execution_date=execution_date)
        # The existing augmenters are deliberately label-free in implementation.
        # PitFeatureRow exposes only the fields they consume, making accidental
        # outcome access impossible at this runtime boundary.
        augmented = augment_system_context_features(rows)  # type: ignore[arg-type]
        current = tuple(
            row for row in augmented if row.execution_date == execution_date
        )
        candidates = tuple(
            FeatureSnapshot(
                candidate_id=row.event_id,
                event_id=row.event_id,
                company_id=row.company_id,
                security_id=row.security_id,
                decision_time=row.decision_time,
                execution_date=row.execution_date,
                features=row.features,
            )
            for row in current
        )
        return PitCandidateAssembly(
            as_of=cutoff,
            execution_date=execution_date,
            trigger_event_count=len(triggers),
            affected_company_count=len(affected_companies),
            context_opportunity_count=len(rows),
            candidate_count=len(candidates),
            candidates=candidates,
        )

    def _base_rows(
        self,
        history: tuple[Event, ...],
        *,
        execution_date: date,
    ) -> tuple[PitFeatureRow, ...]:
        grouped_history: dict[str, list[Event]] = {}
        for event in history:
            if event.company_id:
                grouped_history.setdefault(event.company_id, []).append(event)
        history_by_company = {
            company_id: CompanyEventIndex(company_id, events)
            for company_id, events in grouped_history.items()
        }

        by_company_day: dict[tuple[str, date], list[Event]] = {}
        for trigger in history:
            if trigger.event_type not in _TRIGGER_TYPES or not trigger.company_id:
                continue
            key = (trigger.company_id, decision_market_date(trigger.public_time))
            by_company_day.setdefault(key, []).append(trigger)

        by_execution_session: dict[tuple[str, date], list[_OpportunitySlice]] = {}
        for key in sorted(by_company_day):
            day_triggers = tuple(
                sorted(
                    by_company_day[key],
                    key=lambda event: (event.public_time, event.event_id),
                )
            )
            anchor = day_triggers[-1]
            snapshot = self._resolve_snapshot(anchor, execution_date=execution_date)
            if snapshot is None or snapshot.execution_date > execution_date:
                continue
            session_key = (anchor.company_id, snapshot.execution_date)
            by_execution_session.setdefault(session_key, []).append(
                _OpportunitySlice(snapshot=snapshot, triggers=day_triggers)
            )

        rows: list[PitFeatureRow] = []
        for session_key in sorted(by_execution_session):
            slices = by_execution_session[session_key]
            session_triggers = tuple(
                sorted(
                    (
                        trigger
                        for slice_ in slices
                        for trigger in slice_.triggers
                    ),
                    key=lambda event: (event.public_time, event.event_id),
                )
            )
            if not session_triggers:
                continue
            latest_slice = max(
                slices,
                key=lambda slice_: (
                    slice_.snapshot.decision_time,
                    slice_.snapshot.event_id,
                ),
            )
            snapshot = latest_slice.snapshot
            decision_time = session_triggers[-1].public_time
            if snapshot.decision_time != decision_time:
                # Multiple publication days can merge into one executable session.
                # Rebuild from the latest trigger so the market snapshot and event
                # features share exactly the same PIT decision timestamp.
                rebuilt = self._resolve_snapshot(
                    session_triggers[-1],
                    execution_date=snapshot.execution_date,
                )
                if rebuilt is None:
                    continue
                snapshot = rebuilt
            if snapshot.decision_time != decision_time:
                raise ValueError("opportunity snapshot does not match latest trigger time")

            company_id = session_key[0]
            company_history = history_by_company.get(company_id)
            if company_history is None:
                raise ValueError(f"missing temporal event index for {company_id}")
            features = {
                **snapshot.market_features,
                **build_insider_features(
                    company_history,
                    company_id=company_id,
                    decision_time=decision_time,
                ),
                **build_alternative_features(
                    company_history,
                    company_id=company_id,
                    decision_time=decision_time,
                ),
                **build_congress_features(
                    company_history,
                    company_id=company_id,
                    decision_time=decision_time,
                    enabled=self.enable_congress_features,
                ),
                **build_opportunity_trigger_features(session_triggers),
            }
            features.update(build_research_interactions(features))
            rows.append(
                PitFeatureRow(
                    event_id=_opportunity_id(company_id, snapshot.execution_date),
                    company_id=company_id,
                    security_id=snapshot.security_id,
                    decision_time=decision_time,
                    execution_date=snapshot.execution_date,
                    features=features,
                    trigger_event_ids=tuple(
                        trigger.event_id for trigger in session_triggers
                    ),
                )
            )
        return tuple(rows)

    def _resolve_snapshot(
        self,
        event: Event,
        *,
        execution_date: date,
    ) -> CandidateSnapshot | None:
        try:
            return self.snapshot_builder.build(event)
        except ValueError:
            pass

        publication_day = decision_market_date(event.public_time)
        if publication_day >= execution_date:
            return None
        try:
            return self.snapshot_builder.build_for_execution_date(event, execution_date)
        except ValueError:
            return None


class EventBatchPitCandidateSource:
    """CandidateSource adapter for a durable/current trigger batch.

    Event polling and exchange-session resolution are explicit dependencies. The
    source itself only joins the new trigger batch to existing normalized history
    and deterministic market features.
    """

    def __init__(
        self,
        *,
        event_store: DuckDbEventStore,
        assembler: PitCandidateAssembler,
        trigger_provider: TriggerEventBatchProvider,
        session_resolver: ExecutionSessionResolver,
    ) -> None:
        self.event_store = event_store
        self.assembler = assembler
        self.trigger_provider = trigger_provider
        self.session_resolver = session_resolver
        self.last_assembly: PitCandidateAssembly | None = None

    def candidates(self, as_of: datetime) -> tuple[FeatureSnapshot, ...]:
        cutoff = as_utc(as_of)
        triggers = self.trigger_provider.events(cutoff)
        company_ids = tuple(
            sorted({event.company_id for event in triggers if event.company_id})
        )
        history = (
            self.event_store.all_events(company_ids=company_ids)
            if company_ids
            else ()
        )
        execution_date = self.session_resolver.execution_date(cutoff)
        assembly = self.assembler.assemble(
            triggers,
            all_events=history,
            as_of=cutoff,
            execution_date=execution_date,
        )
        self.last_assembly = assembly
        return assembly.candidates


def _opportunity_id(company_id: str, execution_date: date) -> str:
    # Must stay byte-for-byte compatible with TrainingDatasetBuilder candidate IDs.
    return f"opportunity:{company_id}:{execution_date.isoformat()}"
