from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from stock_trading.core import Source
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.market import (
    ConservativeTiingoResolver,
    DuckDbMarketStore,
    IssuerObservation,
    MarketBackfillService,
    ResolutionStatus,
    SecurityMapping,
    SecurityResolution,
    TiingoClient,
    TiingoNormalizer,
    normalize_tiingo_ticker,
)
from stock_trading.market.execution_time import decision_market_date
from stock_trading.sec import Form4XmlParser
from stock_trading.storage import FileRawStore

from .event_intake import PendingTrigger


@dataclass(frozen=True, slots=True)
class CurrentMarketResolutionFailure:
    company_id: str
    cik: str
    ticker: str
    issuer_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class CurrentMarketSyncResult:
    sync_end_date: date
    selected_event_count: int
    accession_count: int
    company_count: int
    tickers: tuple[str, ...]
    metadata_refreshed: int
    resolved_companies: int
    unresolved_companies: int
    downloaded_price_series: int
    failed_price_series: int
    reused_price_responses: int
    skipped_complete_price_series: int
    benchmark_downloaded: bool
    benchmark_bars_added: int
    failures: tuple[CurrentMarketResolutionFailure, ...]

    @property
    def ready(self) -> bool:
        return self.unresolved_companies == 0 and self.failed_price_series == 0


class _CurrentCoverageTiingoResolver:
    """Resolve a current SEC issuer using completed-session Tiingo coverage.

    Tiingo EOD metadata defines ``endDate`` as the latest date for which Tiingo
    has price data, not as a delisting or security-validity date. For a current
    filing after today's open it is therefore normal for an active ticker's fresh
    metadata to end at the previous completed session.

    Historical resolution remains conservative and unchanged. At this current
    boundary we first require fresh provider coverage through ``coverage_through``.
    We then reuse the historical ticker/name/start-date checks at a date already
    covered by Tiingo and publish an open-ended mapping only after those checks
    pass. Provider coverage that lags the completed exchange session still fails
    closed.
    """

    def __init__(self, coverage_through: date) -> None:
        self.coverage_through = coverage_through
        self._historical = ConservativeTiingoResolver()

    def resolve(
        self,
        observation: IssuerObservation,
        *,
        tiingo_ticker: str,
        tiingo_name: str,
        tiingo_start: date,
        tiingo_end: date | None,
        exchange_code: str | None = None,
    ) -> SecurityResolution:
        if tiingo_end is None:
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                observation,
                None,
                "tiingo_has_no_price_coverage",
            )
        if tiingo_end < self.coverage_through:
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                observation,
                None,
                "tiingo_history_lags_completed_session",
            )

        covered_observation = IssuerObservation(
            sec_cik=observation.sec_cik,
            issuer_name=observation.issuer_name,
            ticker=observation.ticker,
            observed_date=min(observation.observed_date, tiingo_end),
        )
        validated = self._historical.resolve(
            covered_observation,
            tiingo_ticker=tiingo_ticker,
            tiingo_name=tiingo_name,
            tiingo_start=tiingo_start,
            tiingo_end=tiingo_end,
            exchange_code=exchange_code,
        )
        if not validated.resolved or validated.mapping is None:
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                observation,
                None,
                validated.reason,
            )

        mapping = SecurityMapping(
            company_id=validated.mapping.company_id,
            security_id=validated.mapping.security_id,
            ticker=validated.mapping.ticker,
            exchange_code=validated.mapping.exchange_code,
            valid_from=validated.mapping.valid_from,
            valid_to=None,
        )
        return SecurityResolution(
            ResolutionStatus.RESOLVED,
            observation,
            mapping,
            "current_ticker_name_match_with_completed_session_coverage",
        )


def sync_pending_current_market(
    pending: tuple[PendingTrigger, ...],
    *,
    data_root: str | Path,
    market_store: DuckDbMarketStore,
    benchmark_security_id: str,
    tiingo_client: TiingoClient,
    sync_end_date: date,
) -> CurrentMarketSyncResult:
    """Refresh only the market identities/series required by one pending batch.

    Current Form 4 XML is the issuer-identity source. Tiingo metadata is fetched
    fresh on every actionable batch before resolution. ``endDate`` is interpreted
    as a price-history coverage watermark, never as a current security-validity
    boundary. Price data is then extended only through the last completed XNYS
    session supplied by the caller.
    """

    if not pending:
        return CurrentMarketSyncResult(
            sync_end_date=sync_end_date,
            selected_event_count=0,
            accession_count=0,
            company_count=0,
            tickers=(),
            metadata_refreshed=0,
            resolved_companies=0,
            unresolved_companies=0,
            downloaded_price_series=0,
            failed_price_series=0,
            reused_price_responses=0,
            skipped_complete_price_series=0,
            benchmark_downloaded=False,
            benchmark_bars_added=0,
            failures=(),
        )

    data_root = Path(data_root)
    raw_store = FileRawStore(data_root / "raw")
    parser = Form4XmlParser()

    by_accession: dict[tuple[str, str], list[PendingTrigger]] = {}
    for item in pending:
        by_accession.setdefault((item.cik, item.accession_number), []).append(item)

    identities_by_company: dict[str, list[tuple[PendingTrigger, object]]] = {}
    for (cik, accession_number), items in sorted(by_accession.items()):
        raw = raw_store.latest(Source.SEC_EDGAR, accession_number)
        if raw is None:
            raise RuntimeError(
                f"current Form 4 raw artifact is missing for {accession_number}"
            )
        if raw.content_type != "application/xml":
            raise RuntimeError(
                f"current Form 4 raw artifact is not verified XML: {accession_number}"
            )
        identity = parser.issuer_identity(raw)
        normalized_cik = str(cik).strip().lstrip("0").zfill(10)
        if identity.cik != normalized_cik:
            raise ValueError(
                f"Form 4 issuer CIK changed for pending accession {accession_number}"
            )
        company_id = company_id_from_sec_cik(identity.cik)
        if any(item.company_id != company_id for item in items):
            raise ValueError(
                f"Form 4 company identity changed for pending accession {accession_number}"
            )
        anchor = max(items, key=lambda item: (item.public_time, item.event_id))
        identities_by_company.setdefault(company_id, []).append((anchor, identity))

    observations: list[IssuerObservation] = []
    for company_id, values in sorted(identities_by_company.items()):
        tickers = {normalize_tiingo_ticker(identity.ticker) for _, identity in values}
        if len(tickers) != 1:
            raise ValueError(
                f"pending Form 4 batch contains multiple issuer tickers for {company_id}: "
                f"{sorted(tickers)}"
            )
        anchor, identity = max(
            values,
            key=lambda item: (item[0].public_time, item[0].event_id),
        )
        observations.append(
            IssuerObservation(
                sec_cik=identity.cik,
                issuer_name=identity.name,
                ticker=identity.ticker,
                observed_date=decision_market_date(anchor.public_time),
            )
        )

    normalized_tickers = tuple(
        sorted({normalize_tiingo_ticker(item.ticker) for item in observations})
    )
    # Force fresh mutable provider metadata into immutable raw storage. The normal
    # historical backfill remains cache-friendly; only this current boundary does
    # an explicit metadata refresh.
    for ticker in normalized_tickers:
        raw_store.put(tiingo_client.fetch_metadata(ticker))

    market_start = min(item.observed_date for item in observations) - timedelta(days=400)
    market_result = MarketBackfillService(
        client=tiingo_client,
        raw_store=raw_store,
        market_store=market_store,
        resolver=_CurrentCoverageTiingoResolver(sync_end_date),
    ).backfill(
        tuple(observations),
        start=market_start,
        end=sync_end_date,
    )

    failures = tuple(
        CurrentMarketResolutionFailure(
            company_id=resolution.observation.company_id,
            cik=resolution.observation.sec_cik,
            ticker=resolution.observation.ticker,
            issuer_name=resolution.observation.issuer_name,
            reason=resolution.reason,
        )
        for resolution in market_result.resolutions
        if not resolution.resolved
    )

    benchmark_downloaded, benchmark_bars_added = _sync_benchmark(
        tiingo_client,
        raw_store=raw_store,
        market_store=market_store,
        benchmark_security_id=benchmark_security_id,
        sync_end_date=sync_end_date,
    )

    unresolved_companies = len({item.company_id for item in failures})
    return CurrentMarketSyncResult(
        sync_end_date=sync_end_date,
        selected_event_count=len(pending),
        accession_count=len(by_accession),
        company_count=len(identities_by_company),
        tickers=normalized_tickers,
        metadata_refreshed=len(normalized_tickers),
        resolved_companies=market_result.resolved_companies,
        unresolved_companies=unresolved_companies,
        downloaded_price_series=market_result.downloaded_price_series,
        failed_price_series=market_result.failed_price_series,
        reused_price_responses=market_result.reused_price_responses,
        skipped_complete_price_series=market_result.skipped_complete_price_series,
        benchmark_downloaded=benchmark_downloaded,
        benchmark_bars_added=benchmark_bars_added,
        failures=failures,
    )


def _sync_benchmark(
    client: TiingoClient,
    *,
    raw_store: FileRawStore,
    market_store: DuckDbMarketStore,
    benchmark_security_id: str,
    sync_end_date: date,
) -> tuple[bool, int]:
    ticker = "SPY"
    bounds = market_store.date_bounds(benchmark_security_id, ticker)
    if bounds is not None and bounds[1] >= sync_end_date:
        return False, 0

    start = (
        bounds[1] + timedelta(days=1)
        if bounds is not None
        else sync_end_date - timedelta(days=400)
    )
    if start > sync_end_date:
        return False, 0

    record_id = f"prices:{ticker}:{start.isoformat()}:{sync_end_date.isoformat()}"
    raw = raw_store.latest(Source.TIINGO, record_id)
    downloaded = False
    if raw is None:
        raw = client.fetch_prices(ticker, start, sync_end_date)
        raw_store.put(raw)
        downloaded = True
    bars = TiingoNormalizer().parse_prices(
        raw,
        security_id=benchmark_security_id,
        ticker=ticker,
    )
    before = market_store.count_bars(
        benchmark_security_id,
        ticker,
        start,
        sync_end_date,
    )
    market_store.put_many(bars)
    after = market_store.count_bars(
        benchmark_security_id,
        ticker,
        start,
        sync_end_date,
    )
    return downloaded, max(0, after - before)
