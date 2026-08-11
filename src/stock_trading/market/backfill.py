from dataclasses import dataclass
from datetime import date

from stock_trading.storage import FileRawStore

from .normalize import TiingoNormalizer
from .resolution import ConservativeTiingoResolver, IssuerObservation, SecurityResolution
from .security import SecurityRegistry
from .store import DuckDbMarketStore


@dataclass(frozen=True, slots=True)
class MarketBackfillResult:
    resolutions: tuple[SecurityResolution, ...]
    resolved_companies: int
    unresolved_observations: int
    downloaded_price_series: int
    normalized_bars: int


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

        metadata_cache = {}
        resolutions: list[SecurityResolution] = []
        series_to_fetch: dict[tuple[str, str], tuple[date, date]] = {}

        for observation in observations:
            ticker = observation.ticker.strip().upper().replace(".", "-")
            metadata = metadata_cache.get(ticker)
            if metadata is None:
                raw_metadata = self.client.fetch_metadata(ticker)
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
        for (company_id, ticker), (fetch_start, fetch_end) in sorted(series_to_fetch.items()):
            raw_prices = self.client.fetch_prices(ticker, fetch_start, fetch_end)
            self.raw_store.put(raw_prices)
            bars = self.normalizer.parse_prices(
                raw_prices,
                company_id=company_id,
                ticker=ticker,
            )
            self.market_store.put_many(bars)
            normalized_bars += len(bars)

        unresolved = sum(1 for resolution in resolutions if not resolution.resolved)
        return MarketBackfillResult(
            resolutions=tuple(resolutions),
            resolved_companies=len(series_to_fetch),
            unresolved_observations=unresolved,
            downloaded_price_series=len(series_to_fetch),
            normalized_bars=normalized_bars,
        )
