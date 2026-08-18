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
    security_id: str
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
        benchmark_security_id: str | None = None,
        benchmark_company_id: str | None = None,
        feature_lookback_bars: int = 260,
        label_horizons: tuple[int, ...] = (1, 5, 20, 60),
    ) -> None:
        if feature_lookback_bars <= 0:
            raise ValueError("feature_lookback_bars must be > 0")
        if not label_horizons or any(horizon <= 0 for horizon in label_horizons):
            raise ValueError("label_horizons must contain positive horizons")
        if (
            benchmark_security_id is not None
            and benchmark_company_id is not None
            and benchmark_security_id != benchmark_company_id
        ):
            raise ValueError("benchmark security aliases disagree")
        benchmark = benchmark_security_id or benchmark_company_id
        if not benchmark:
            raise ValueError("benchmark_security_id is required")

        self.market_store = market_store
        self.benchmark_security_id = benchmark
        # Compatibility attribute for the existing experiment config. It is no
        # longer interpreted as a legal-company identity.
        self.benchmark_company_id = benchmark
        self.feature_lookback_bars = feature_lookback_bars
        self.label_horizons = tuple(sorted(set(label_horizons)))

    def build(self, event: Event) -> CandidateSnapshot:
        if not event.company_id:
            raise ValueError("candidate event must have a canonical company_id")

        decision_day = decision_market_date(event.public_time)
        security_id = self.market_store.security_for_company(event.company_id, decision_day)
        if security_id is None:
            raise ValueError("no verified security mapping available for candidate company")

        execution_bar = self.market_store.next_bar_after(security_id, decision_day)
        if execution_bar is None:
            raise ValueError("no future market bar available for candidate security")

        benchmark_execution_bar = self.market_store.bar_on(
            self.benchmark_security_id,
            execution_bar.date,
        )
        if benchmark_execution_bar is None:
            raise ValueError("benchmark has no bar on candidate execution date")

        stock_history = self.market_store.bars_before(
            security_id,
            decision_day,
            self.feature_lookback_bars,
        )
        benchmark_history = self.market_store.bars_before(
            self.benchmark_security_id,
            decision_day,
            self.feature_lookback_bars,
        )
        if not stock_history:
            raise ValueError("candidate security has no completed market history before publication")
        if not benchmark_history:
            raise ValueError("benchmark has no completed market history before publication")

        return CandidateSnapshot(
            event_id=event.event_id,
            company_id=event.company_id,
            security_id=security_id,
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

    def build_for_execution_date(
        self,
        event: Event,
        execution_date: date,
    ) -> CandidateSnapshot:
        """Build a live/PAPER snapshot for an externally resolved future session.

        Unlike :meth:`build`, this method never requires an execution-day market
        bar. The execution date must come from a market-calendar/session boundary
        outside the model feature builder. All model features use completed bars
        strictly before the event's publication market date.

        If stored market history already proves that another trading session
        occurred between publication and ``execution_date``, fail closed rather
        than silently allowing a caller to shift the opportunity to a later open.
        """

        if not event.company_id:
            raise ValueError("candidate event must have a canonical company_id")
        decision_day = decision_market_date(event.public_time)
        if execution_date <= decision_day:
            raise ValueError("execution_date must be after publication market date")

        security_id = self.market_store.security_for_company(event.company_id, decision_day)
        if security_id is None:
            raise ValueError("no verified security mapping available for candidate company")

        known_stock_next = self.market_store.next_bar_after(security_id, decision_day)
        if known_stock_next is not None and known_stock_next.date < execution_date:
            raise ValueError("execution_date skips a known candidate trading session")
        known_benchmark_next = self.market_store.next_bar_after(
            self.benchmark_security_id,
            decision_day,
        )
        if known_benchmark_next is not None and known_benchmark_next.date < execution_date:
            raise ValueError("execution_date skips a known benchmark trading session")

        stock_history = self.market_store.bars_before(
            security_id,
            decision_day,
            self.feature_lookback_bars,
        )
        benchmark_history = self.market_store.bars_before(
            self.benchmark_security_id,
            decision_day,
            self.feature_lookback_bars,
        )
        if not stock_history:
            raise ValueError("candidate security has no completed market history before publication")
        if not benchmark_history:
            raise ValueError("benchmark has no completed market history before publication")

        execution_ticker = (
            known_stock_next.ticker
            if known_stock_next is not None and known_stock_next.date == execution_date
            else stock_history[-1].ticker
        )
        return CandidateSnapshot(
            event_id=event.event_id,
            company_id=event.company_id,
            security_id=security_id,
            decision_time=event.public_time,
            decision_market_date=decision_day,
            execution_date=execution_date,
            first_tradable_time=conservative_first_tradable_time(
                event.public_time,
                execution_date,
            ),
            execution_ticker=execution_ticker,
            market_features=build_market_features(stock_history, benchmark_history),
        )

    def label(self, snapshot: CandidateSnapshot) -> LabeledCandidate:
        max_horizon = max(self.label_horizons)
        stock_future = self.market_store.bars_from(
            snapshot.security_id,
            snapshot.execution_date,
            max_horizon,
        )
        benchmark_future = self.market_store.bars_from(
            self.benchmark_security_id,
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
