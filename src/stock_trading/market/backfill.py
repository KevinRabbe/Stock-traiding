from dataclasses import dataclass
from datetime import date, timedelta

import httpx

from stock_trading.core import Source
from stock_trading.storage import FileRawStore

from .normalize import TiingoNormalizer
from .resolution import (
    ConservativeTiingoResolver,
    IssuerObservation,
    ResolutionStatus,
    SecurityResolution,
)
from .security import SecurityRegistry
from .store import DuckDbMarketStore
from .tiingo import normalize_tiingo_ticker


@dataclass(frozen=True, slots=True)
class MarketBackfillResult:
    resolutions: tuple[SecurityResolution, ...]
    resolved_companies: int
    unresolved_observations: int
    downloaded_price_series: int
    normalized_bars: int
    failed_metadata_requests: int = 0
    failed_price_series: int = 0
    reused_metadata_responses: int = 0
    reused_price_responses: int = 0
    skipped_complete_price_series: int = 0


class MarketBackfillService:
    """Resolve SEC issuer observations and backfill only verified Tiingo histories."""

    def __init__(
        self,
        *,
        client,
        raw_store: FileRawStore,
        market_store: DuckDbMarketStore,
        security_registry: SecurityRegistry | None = None,
        normalizer: TiingoNormalizer | None = None,
        resolver: ConservativeTiingoResolver | None = None,
    ) -> None:
        self.client = client
        self.raw_store = raw_store
        self.market_store = market_store
        self.security_registry = security_registry or SecurityRegistry()
        self.normalizer = normalizer or TiingoNormalizer()
        self.resolver = resolver or ConservativeTiingoResolver()

    def backfill(
        self,
        observations: tuple[IssuerObservation, ...] | list[IssuerObservation],
        *,
        start: date,
        end: date,
    ) -> MarketBackfillResult:
        if end < start:
            raise ValueError("end must be >= start")

        metadata_cache: dict[str, object | None] = {}
        metadata_failures: dict[str, str] = {}
        resolutions: list[SecurityResolution] = []
        series_to_fetch: dict[tuple[str, str], tuple[date, date]] = {}
        failed_metadata_requests = 0
        reused_metadata_responses = 0

        for observation in observations:
            try:
                ticker = normalize_tiingo_ticker(observation.ticker)
            except ValueError:
                resolutions.append(
                    SecurityResolution(
                        ResolutionStatus.UNRESOLVED,
                        observation,
                        None,
                        "unsupported_ticker",
                    )
                )
                continue

            if ticker in metadata_failures:
                resolutions.append(
                    SecurityResolution(
                        ResolutionStatus.UNRESOLVED,
                        observation,
                        None,
                        metadata_failures[ticker],
                    )
                )
                continue

            metadata = metadata_cache.get(ticker)
            if metadata is None:
                metadata_record_id = f"metadata:{ticker}"
                raw_metadata = self.raw_store.latest(Source.TIINGO, metadata_record_id)
                if raw_metadata is None:
                    try:
                        raw_metadata = self.client.fetch_metadata(ticker)
                    except httpx.HTTPError as exc:
                        failed_metadata_requests += 1
                        reason = _http_failure_reason("tiingo_metadata", exc)
                        metadata_failures[ticker] = reason
                        resolutions.append(
                            SecurityResolution(
                                ResolutionStatus.UNRESOLVED,
                                observation,
                                None,
                                reason,
                            )
                        )
                        continue
                    self.raw_store.put(raw_metadata)
                else:
                    reused_metadata_responses += 1

                try:
                    metadata = self.normalizer.parse_metadata(raw_metadata)
                except (ValueError, KeyError, TypeError) as exc:
                    reason = _metadata_parse_failure_reason(exc)
                    metadata_failures[ticker] = reason
                    resolutions.append(
                        SecurityResolution(
                            ResolutionStatus.UNRESOLVED,
                            observation,
                            None,
                            reason,
                        )
                    )
                    continue
                metadata_cache[ticker] = metadata

            resolution = self.resolver.resolve(
                observation,
                tiingo_ticker=metadata.ticker,
                tiingo_name=metadata.name,
                tiingo_start=metadata.start_date,
                tiingo_end=metadata.end_date,
                exchange_code=metadata.exchange_code,
            )
            resolutions.append(resolution)
            if not resolution.resolved or resolution.mapping is None:
                continue

            self.security_registry.add(resolution.mapping)
            mapping = resolution.mapping
            fetch_start = max(start, mapping.valid_from)
            fetch_end = min(end, mapping.valid_to) if mapping.valid_to is not None else end
            if fetch_end < fetch_start:
                continue

            key = (mapping.company_id, mapping.ticker)
            existing = series_to_fetch.get(key)
            if existing is None:
                series_to_fetch[key] = (fetch_start, fetch_end)
            else:
                series_to_fetch[key] = (min(existing[0], fetch_start), max(existing[1], fetch_end))

        resolutions = list(_promote_validated_company_aliases(resolutions))

        normalized_bars = 0
        downloaded_price_series = 0
        failed_price_series = 0
        reused_price_responses = 0
        skipped_complete_price_series = 0

        for (company_id, ticker), (fetch_start, fetch_end) in sorted(series_to_fetch.items()):
            full_record_id = _price_record_id(ticker, fetch_start, fetch_end)
            raw_full = self.raw_store.latest(Source.TIINGO, full_record_id)
            if raw_full is not None:
                reused_price_responses += 1
                bars = self.normalizer.parse_prices(
                    raw_full,
                    company_id=company_id,
                    ticker=ticker,
                )
                self.market_store.put_many(bars)
            else:
                stored_bounds = self.market_store.date_bounds(company_id, ticker)
                missing_ranges = _missing_date_ranges(fetch_start, fetch_end, stored_bounds)
                if not missing_ranges:
                    skipped_complete_price_series += 1

                for range_start, range_end in missing_ranges:
                    record_id = _price_record_id(ticker, range_start, range_end)
                    raw_prices = self.raw_store.latest(Source.TIINGO, record_id)
                    if raw_prices is None:
                        try:
                            raw_prices = self.client.fetch_prices(ticker, range_start, range_end)
                        except httpx.HTTPError:
                            failed_price_series += 1
                            continue
                        self.raw_store.put(raw_prices)
                        downloaded_price_series += 1
                    else:
                        reused_price_responses += 1

                    bars = self.normalizer.parse_prices(
                        raw_prices,
                        company_id=company_id,
                        ticker=ticker,
                    )
                    self.market_store.put_many(bars)

            normalized_bars += self.market_store.count_bars(
                company_id,
                ticker,
                fetch_start,
                fetch_end,
            )

        unresolved = sum(1 for resolution in resolutions if not resolution.resolved)
        return MarketBackfillResult(
            resolutions=tuple(resolutions),
            resolved_companies=len(series_to_fetch),
            unresolved_observations=unresolved,
            downloaded_price_series=downloaded_price_series,
            normalized_bars=normalized_bars,
            failed_metadata_requests=failed_metadata_requests,
            failed_price_series=failed_price_series,
            reused_metadata_responses=reused_metadata_responses,
            reused_price_responses=reused_price_responses,
            skipped_complete_price_series=skipped_complete_price_series,
        )


def _promote_validated_company_aliases(
    resolutions: list[SecurityResolution] | tuple[SecurityResolution, ...],
) -> tuple[SecurityResolution, ...]:
    """Accept a name variant only after the same SEC CIK+ticker has an exact match.

    This is deliberately narrower than fuzzy name matching. A mismatched or
    placeholder SEC issuer name is promoted only when another observation for
    the same canonical SEC company and normalized ticker already passed the
    ticker, date and company-name checks against the same Tiingo mapping.
    """

    validated = {
        (resolution.mapping.company_id, resolution.mapping.ticker): resolution.mapping
        for resolution in resolutions
        if resolution.resolved and resolution.mapping is not None
    }
    promoted: list[SecurityResolution] = []
    for resolution in resolutions:
        if resolution.reason != "company_name_mismatch":
            promoted.append(resolution)
            continue

        observation = resolution.observation
        try:
            ticker = normalize_tiingo_ticker(observation.ticker)
        except ValueError:
            promoted.append(resolution)
            continue
        mapping = validated.get((observation.company_id, ticker))
        if mapping is None or not mapping.contains(observation.observed_date):
            promoted.append(resolution)
            continue

        promoted.append(
            SecurityResolution(
                ResolutionStatus.RESOLVED,
                observation,
                mapping,
                "sec_cik_ticker_validated_by_matching_alias",
            )
        )
    return tuple(promoted)


def _price_record_id(ticker: str, start: date, end: date) -> str:
    return f"prices:{ticker}:{start.isoformat()}:{end.isoformat()}"


def _missing_date_ranges(
    start: date,
    end: date,
    stored_bounds: tuple[date, date] | None,
) -> tuple[tuple[date, date], ...]:
    if stored_bounds is None:
        return ((start, end),)

    stored_start, stored_end = stored_bounds
    missing: list[tuple[date, date]] = []
    if start < stored_start:
        prefix_end = min(end, stored_start - timedelta(days=1))
        if start <= prefix_end:
            missing.append((start, prefix_end))
    if end > stored_end:
        suffix_start = max(start, stored_end + timedelta(days=1))
        if suffix_start <= end:
            missing.append((suffix_start, end))
    return tuple(missing)


def _http_failure_reason(prefix: str, exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{prefix}_http_{exc.response.status_code}"
    return f"{prefix}_request_error"


def _metadata_parse_failure_reason(exc: Exception) -> str:
    message = str(exc).lower()
    if "no startdate" in message:
        return "tiingo_metadata_missing_start_date"
    if "no company name" in message:
        return "tiingo_metadata_missing_company_name"
    if isinstance(exc, KeyError) and "ticker" in message:
        return "tiingo_metadata_missing_ticker"
    return "tiingo_metadata_invalid"
