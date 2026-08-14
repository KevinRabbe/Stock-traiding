from dataclasses import dataclass
from datetime import date, datetime
from math import nan
from typing import Iterable

import numpy as np

from stock_trading.core import Event, EventType
from stock_trading.extraction import ALLOWED_TOPICS
from stock_trading.features import (
    build_alternative_features,
    build_congress_features,
    build_insider_features,
)
from stock_trading.market import CandidateSnapshot, CandidateSnapshotBuilder, decision_market_date


_TRIGGER_TYPES = (
    EventType.INSIDER_TRANSACTION,
    EventType.GOVERNMENT_CONTRACT,
    EventType.LOBBYING_ACTIVITY,
)


@dataclass(frozen=True, slots=True)
class TrainingRow:
    event_id: str
    company_id: str
    decision_time: datetime
    execution_date: date
    exit_date_20d: date
    features: dict[str, float | None]
    stock_return_20d: float
    benchmark_return_20d: float
    alpha_20d: float
    downside_20d: float
    mfe_20d: float
    positive_alpha_20d: int
    trigger_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    names: tuple[str, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[TrainingRow]) -> "FeatureSchema":
        names = sorted({name for row in rows for name in row.features})
        if not names:
            raise ValueError("cannot build a feature schema from empty features")
        return cls(tuple(names))

    def vector(self, features: dict[str, float | None]) -> list[float]:
        return [
            float(features[name]) if name in features and features[name] is not None else nan
            for name in self.names
        ]

    def matrix(self, rows: Iterable[TrainingRow]) -> np.ndarray:
        materialized = tuple(rows)
        if not materialized:
            return np.empty((0, len(self.names)), dtype=np.float32)
        return np.asarray(
            [self.vector(row.features) for row in materialized],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class _OpportunitySlice:
    snapshot: CandidateSnapshot
    triggers: tuple[Event, ...]


class TrainingDatasetBuilder:
    """Create one model row per company/execution session without crossing PIT."""

    def __init__(
        self,
        snapshot_builder: CandidateSnapshotBuilder,
        *,
        positive_alpha_threshold: float = 0.02,
        target_horizon: int = 20,
        enable_congress_features: bool = False,
    ) -> None:
        if target_horizon != 20:
            raise ValueError("V1 TrainingRow fields are fixed to a 20-day target")
        self.snapshot_builder = snapshot_builder
        self.positive_alpha_threshold = positive_alpha_threshold
        self.target_horizon = target_horizon
        self.enable_congress_features = enable_congress_features

    def build(
        self,
        trigger_events: Iterable[Event],
        *,
        all_events: Iterable[Event],
    ) -> tuple[TrainingRow, ...]:
        event_history = tuple(all_events)
        history_by_company: dict[str, tuple[Event, ...]] = {}
        grouped_history: dict[str, list[Event]] = {}
        for event in event_history:
            if event.company_id:
                grouped_history.setdefault(event.company_id, []).append(event)
        for company_id, events in grouped_history.items():
            history_by_company[company_id] = tuple(
                sorted(events, key=lambda event: (event.public_time, event.event_id))
            )

        # Quarterly Form 4 data can contain several transaction lines from one
        # filing. Collapse those raw triggers before touching the market store.
        # Under the conservative daily/EOD policy one company/publication day can
        # create at most one next-session opportunity.
        by_company_day: dict[tuple[str, date], list[Event]] = {}
        for trigger in trigger_events:
            if trigger.event_type not in _TRIGGER_TYPES or not trigger.company_id:
                continue
            key = (trigger.company_id, decision_market_date(trigger.public_time))
            by_company_day.setdefault(key, []).append(trigger)

        # Different publication days can still point to the same next executable
        # session, for example Friday plus weekend disclosures before Monday open.
        # Resolve one anchor per company/day, then merge slices that share the
        # exact company + execution session.
        by_execution_session: dict[tuple[str, date], list[_OpportunitySlice]] = {}
        for key in sorted(by_company_day):
            triggers = tuple(
                sorted(
                    by_company_day[key],
                    key=lambda event: (event.public_time, event.event_id),
                )
            )
            anchor = triggers[-1]
            try:
                snapshot = self.snapshot_builder.build(anchor)
            except ValueError:
                # Unresolved/no-market-history candidates cannot form opportunities.
                continue
            session_key = (anchor.company_id, snapshot.execution_date)
            by_execution_session.setdefault(session_key, []).append(
                _OpportunitySlice(snapshot=snapshot, triggers=triggers)
            )

        rows: list[TrainingRow] = []
        for session_key in sorted(by_execution_session):
            slices = by_execution_session[session_key]
            triggers = tuple(
                sorted(
                    (
                        trigger
                        for slice_ in slices
                        for trigger in slice_.triggers
                    ),
                    key=lambda event: (event.public_time, event.event_id),
                )
            )
            if not triggers:
                continue

            # Use the latest public information that is still known before this
            # execution session. Its snapshot also contains the freshest PIT
            # market state available to the opportunity.
            latest_slice = max(
                slices,
                key=lambda slice_: (
                    slice_.snapshot.decision_time,
                    slice_.snapshot.event_id,
                ),
            )
            snapshot = latest_slice.snapshot
            decision_time = triggers[-1].public_time
            if snapshot.decision_time != decision_time:
                raise ValueError("opportunity snapshot does not match latest trigger time")

            try:
                labeled = self.snapshot_builder.label(snapshot)
            except ValueError:
                # Immature/alignment failures are not valid training samples.
                continue

            label = next(
                (item for item in labeled.labels if item.horizon == self.target_horizon),
                None,
            )
            if label is None:
                continue

            company_id = session_key[0]
            company_history = history_by_company.get(company_id, ())
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
                **build_opportunity_trigger_features(triggers),
            }
            features.update(build_research_interactions(features))

            rows.append(
                TrainingRow(
                    event_id=_opportunity_id(company_id, label.start_date),
                    company_id=company_id,
                    decision_time=decision_time,
                    execution_date=label.start_date,
                    exit_date_20d=label.end_date,
                    features=features,
                    stock_return_20d=label.stock_return,
                    benchmark_return_20d=label.benchmark_return,
                    alpha_20d=label.alpha,
                    downside_20d=max(0.0, -label.max_adverse_excursion),
                    mfe_20d=max(0.0, label.max_favorable_excursion),
                    positive_alpha_20d=int(label.alpha >= self.positive_alpha_threshold),
                    trigger_event_ids=tuple(trigger.event_id for trigger in triggers),
                )
            )

        return tuple(rows)


def build_opportunity_trigger_features(
    events: Iterable[Event],
) -> dict[str, float | None]:
    """Aggregate newly public trigger facts for one executable opportunity."""

    materialized = tuple(
        sorted(events, key=lambda event: (event.public_time, event.event_id))
    )
    if not materialized:
        raise ValueError("opportunity trigger events must not be empty")

    insider_count = sum(
        event.event_type is EventType.INSIDER_TRANSACTION for event in materialized
    )
    contract_count = sum(
        event.event_type is EventType.GOVERNMENT_CONTRACT for event in materialized
    )
    lobbying_count = sum(
        event.event_type is EventType.LOBBYING_ACTIVITY for event in materialized
    )
    values = [
        value
        for event in materialized
        if (value := _trigger_value(event)) is not None
    ]

    features: dict[str, float | None] = {
        "trigger.is_insider": float(insider_count > 0),
        "trigger.is_contract": float(contract_count > 0),
        "trigger.is_lobbying": float(lobbying_count > 0),
        # Backward-compatible name now represents all newly public trigger value
        # in the opportunity instead of one arbitrary source row.
        "trigger.source_value": sum(values) if values else None,
        "trigger.event_count": float(len(materialized)),
        "trigger.insider_event_count": float(insider_count),
        "trigger.contract_event_count": float(contract_count),
        "trigger.lobbying_event_count": float(lobbying_count),
        "trigger.unique_actor_count": float(
            len({event.actor_id for event in materialized if event.actor_id})
        ),
        "trigger.source_value_sum": sum(values) if values else None,
        "trigger.source_value_max": max(values) if values else None,
    }

    semantics = [event.semantic for event in materialized if event.semantic is not None]
    features["trigger.semantic.count"] = float(len(semantics))
    for attribute in (
        "novelty",
        "importance",
        "company_relevance",
        "policy_relevance",
        "confidence",
    ):
        semantic_values = [
            float(value)
            for semantic in semantics
            if (value := getattr(semantic, attribute, None)) is not None
        ]
        features[f"trigger.semantic.{attribute}"] = (
            max(semantic_values) if semantic_values else None
        )

    topics = {
        topic
        for semantic in semantics
        for topic in semantic.topics
    }
    for topic in sorted(ALLOWED_TOPICS):
        feature_name = "trigger.topic." + topic.lower().replace(".", "_")
        features[feature_name] = float(topic in topics)
    return features


def build_trigger_features(event: Event) -> dict[str, float | None]:
    """Compatibility wrapper for callers that still have one trigger event."""

    return build_opportunity_trigger_features((event,))


def build_research_interactions(
    features: dict[str, float | None],
) -> dict[str, float | None]:
    """Explicit interaction features motivated by recent alternative-data evidence."""

    appreciation = features.get("market.appreciation_gt_10pct_20d")
    open_market_buys = features.get("insider.open_market_buy_count_30d")
    cluster_buy = features.get("insider.cluster_buy_30d")
    near_high = features.get("market.within_10pct_252d_high")
    contract_surprise = features.get("contracts.surprise_30d")
    new_lobbying_topics = features.get("lobbying.new_issue_codes_90d")
    relational = features.get("cross.relational_convergence_score")

    return {
        "interaction.insider_buy_after_10pct_appreciation_20d": _and_positive(
            appreciation,
            open_market_buys,
        ),
        "interaction.cluster_buy_near_52w_high": _and_positive(cluster_buy, near_high),
        "interaction.contract_acceleration_plus_new_lobbying_topic": (
            float(contract_surprise > 1.0 and new_lobbying_topics > 0)
            if contract_surprise is not None and new_lobbying_topics is not None
            else None
        ),
        "interaction.multi_source_convergence": (
            float(relational >= 3.0) if relational is not None else None
        ),
    }


def _opportunity_id(company_id: str, execution_date: date) -> str:
    return f"opportunity:{company_id}:{execution_date.isoformat()}"


def _and_positive(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left > 0 and right > 0)


def _trigger_value(event: Event) -> float | None:
    if event.event_type is EventType.INSIDER_TRANSACTION:
        value = getattr(event.payload, "value", None)
    elif event.event_type is EventType.GOVERNMENT_CONTRACT:
        value = getattr(event.payload, "obligation_amount", None)
    elif event.event_type is EventType.LOBBYING_ACTIVITY:
        value = getattr(event.payload, "amount", None)
    else:
        value = None
    return float(value) if value is not None else None
