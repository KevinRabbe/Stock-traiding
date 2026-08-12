from dataclasses import dataclass
from datetime import date, datetime

from stock_trading.core import Event

from .execution_time import conservative_first_tradable_time, decision_market_date
from .features import build_market_features
from .labels import ForwardLabel, build_standard_labels
from .store import DuckDbMarketStore


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    """Point-in-time model input derived from one public sparse event."""

    event_id: str
    company_id: str
    decision_time: datetime
    decision_market_date: date
    execution_date: date
    first_tradable_time: datetime
    execution_ticker: str
    market_features: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class LabeledCandidate:
    """Training-only wrapper that keeps future outcomes separate from inputs."""

    snapshot: CandidateSnapshot
    labels: tuple[ForwardLabel, ...]


class CandidateSnapshotBuilder:
    def __init__(
        self,
        market_store: DuckDbMarketStore,
        *,
        benchmark_company_id: str,
        feature_lookback_bars: int = 260,
        label_horizons: tuple[int, ...] = (1, 5, 20, 60),
    ) -> None:
        if feature_lookback_bars <= 0:
            raise ValueError("feature_lookback_bars must be > 0")
        if not label_horizons or any(horizon <= 0 for horizon in label_horizons):
            raise ValueError("label_horizons must contain positive horizons")

        self.market_store = market_store
        self.benchmark_company_id = benchmark_company_id
        self.feature_lookback_bars = feature_lookback_bars
        self.label_horizons = tuple(sorted(set(label_horizons)))

    def build(self, event: Event) -> CandidateSnapshot:
        if not event.company_id:
            raise ValueError("candidate event must have a canonical company_id")

        decision_day = decision_market_date(event.public_time)
        execution_bar = self.market_store.next_bar_after(event.company_id, decision_day)
        if execution_bar is None:
            raise ValueError("no future market bar available for candidate company")

        benchmark_execution_bar = self.market_store.bar_on(
            self.benchmark_company_id,
            execution_bar.date,
        )
        if benchmark_execution_bar is None:
            raise ValueError("benchmark has no bar on candidate execution date")

        stock_history = self.market_store.bars_before(
            event.company_id,
            decision_day,
            self.feature_lookback_bars,
        )
        benchmark_history = self.market_store.bars_before(
            self.benchmark_company_id,
            decision_day,
            self.feature_lookback_bars,
        )
        if not stock_history:
            raise ValueError("candidate company has no completed market history before publication")
        if not benchmark_history:
            raise ValueError("benchmark has no completed market history before publication")

        return CandidateSnapshot(
            event_id=event.event_id,
            company_id=event.company_id,
            decision_time=event.public_time,
            decision_market_date=decision_day,
            execution_date=execution_bar.date,
            first_tradable_time=conservative_first_tradable_time(
                event.public_time,
                execution_bar.date,
            ),
            execution_ticker=execution_bar.ticker,
            market_features=build_market_features(stock_history, benchmark_history),
        )

    def label(self, snapshot: CandidateSnapshot) -> LabeledCandidate:
        max_horizon = max(self.label_horizons)
        stock_future = self.market_store.bars_from(
            snapshot.company_id,
            snapshot.execution_date,
            max_horizon,
        )
        benchmark_future = self.market_store.bars_from(
            self.benchmark_company_id,
            snapshot.execution_date,
            max_horizon,
        )

        labels = build_standard_labels(
            stock_future,
            benchmark_future,
            horizons=self.label_horizons,
        )
        if not labels:
            raise ValueError("candidate has no mature aligned forward labels")
        return LabeledCandidate(snapshot=snapshot, labels=labels)
