from dataclasses import dataclass
from datetime import date

import httpx

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


@dataclass(frozen=True, slots=True)
class MarketBackfillResult:
    resolutions: tuple[SecurityResolution, ...]
    resolved_companies: int
    unresolved_observations: int
    downloaded_price_series: int
    normalized_bars: int
    failed_metadata_requests: int = 0
    failed_price_series: int = 0


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

        for observation in observations:
            ticker = observation.ticker.strip().upper().replace(".", "-")
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
                metadata = self.normalizer.parse_metadata(raw_metadata)
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

        normalized_bars = 0
        downloaded_price_series = 0
        failed_price_series = 0
        for (company_id, ticker), (fetch_start, fetch_end) in sorted(series_to_fetch.items()):
            try:
                raw_prices = self.client.fetch_prices(ticker, fetch_start, fetch_end)
            except httpx.HTTPError:
                failed_price_series += 1
                continue
            self.raw_store.put(raw_prices)
            bars = self.normalizer.parse_prices(
                raw_prices,
                company_id=company_id,
                ticker=ticker,
            )
            self.market_store.put_many(bars)
            downloaded_price_series += 1
            normalized_bars += len(bars)

        unresolved = sum(1 for resolution in resolutions if not resolution.resolved)
        return MarketBackfillResult(
            resolutions=tuple(resolutions),
            resolved_companies=len(series_to_fetch),
            unresolved_observations=unresolved,
            downloaded_price_series=downloaded_price_series,
            normalized_bars=normalized_bars,
            failed_metadata_requests=failed_metadata_requests,
            failed_price_series=failed_price_series,
        )


def _http_failure_reason(prefix: str, exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{prefix}_http_{exc.response.status_code}"
    return f"{prefix}_request_error"
