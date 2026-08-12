from dataclasses import dataclass
from datetime import date, datetime
from math import nan
from typing import Iterable

import numpy as np

from stock_trading.core import Event, EventType
from stock_trading.extraction import ALLOWED_TOPICS
from stock_trading.features import build_alternative_features, build_insider_features
from stock_trading.market import CandidateSnapshotBuilder


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


class TrainingDatasetBuilder:
    """Create model-ready rows without crossing the point-in-time boundary."""

    def __init__(
        self,
        snapshot_builder: CandidateSnapshotBuilder,
        *,
        positive_alpha_threshold: float = 0.02,
        target_horizon: int = 20,
    ) -> None:
        if target_horizon <= 0:
            raise ValueError("target_horizon must be > 0")
        self.snapshot_builder = snapshot_builder
        self.positive_alpha_threshold = positive_alpha_threshold
        self.target_horizon = target_horizon

    def build(
        self,
        trigger_events: Iterable[Event],
        *,
        all_events: Iterable[Event],
    ) -> tuple[TrainingRow, ...]:
        event_history = tuple(all_events)
        rows: list[TrainingRow] = []

        for trigger in sorted(trigger_events, key=lambda event: (event.public_time, event.event_id)):
            if trigger.event_type not in _TRIGGER_TYPES or not trigger.company_id:
                continue

            try:
                snapshot = self.snapshot_builder.build(trigger)
                labeled = self.snapshot_builder.label(snapshot)
            except ValueError:
                # Unresolved/no-market-history/immature candidates are not valid samples.
                continue

            label = next(
                (item for item in labeled.labels if item.horizon == self.target_horizon),
                None,
            )
            if label is None:
                continue

            features = {
                **snapshot.market_features,
                **build_insider_features(
                    event_history,
                    company_id=trigger.company_id,
                    decision_time=trigger.public_time,
                ),
                **build_alternative_features(
                    event_history,
                    company_id=trigger.company_id,
                    decision_time=trigger.public_time,
                ),
                **build_trigger_features(trigger),
            }
            rows.append(
                TrainingRow(
                    event_id=trigger.event_id,
                    company_id=trigger.company_id,
                    decision_time=trigger.public_time,
                    execution_date=label.start_date,
                    exit_date_20d=label.end_date,
                    features=features,
                    stock_return_20d=label.stock_return,
                    benchmark_return_20d=label.benchmark_return,
                    alpha_20d=label.alpha,
                    downside_20d=max(0.0, -label.max_adverse_excursion),
                    mfe_20d=max(0.0, label.max_favorable_excursion),
                    positive_alpha_20d=int(label.alpha >= self.positive_alpha_threshold),
                )
            )

        return tuple(rows)


def build_trigger_features(event: Event) -> dict[str, float | None]:
    features: dict[str, float | None] = {
        "trigger.is_insider": float(event.event_type is EventType.INSIDER_TRANSACTION),
        "trigger.is_contract": float(event.event_type is EventType.GOVERNMENT_CONTRACT),
        "trigger.is_lobbying": float(event.event_type is EventType.LOBBYING_ACTIVITY),
        "trigger.source_value": _trigger_value(event),
    }

    semantic = event.semantic
    features["trigger.semantic.novelty"] = semantic.novelty if semantic else None
    features["trigger.semantic.importance"] = semantic.importance if semantic else None
    features["trigger.semantic.company_relevance"] = (
        semantic.company_relevance if semantic else None
    )
    features["trigger.semantic.policy_relevance"] = semantic.policy_relevance if semantic else None
    features["trigger.semantic.confidence"] = semantic.confidence if semantic else None

    topics = set(semantic.topics) if semantic else set()
    for topic in sorted(ALLOWED_TOPICS):
        feature_name = "trigger.topic." + topic.lower().replace(".", "_")
        features[feature_name] = float(topic in topics)
    return features


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
