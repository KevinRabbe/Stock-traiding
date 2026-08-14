import csv
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import MarketBar, SecurityMapping


_MARKET_COLUMNS = (
    "security_id",
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
    "dividend_cash",
    "split_factor",
)
_BULK_UPSERT_SQL = """
    INSERT INTO security_market_daily (
        security_id, ticker, date,
        open, high, low, close, volume,
        adj_open, adj_high, adj_low, adj_close, adj_volume,
        dividend_cash, split_factor
    )
    SELECT
        security_id,
        ticker,
        CAST(date AS DATE),
        CAST(open AS DOUBLE),
        CAST(high AS DOUBLE),
        CAST(low AS DOUBLE),
        CAST(close AS DOUBLE),
        CAST(volume AS DOUBLE),
        CAST(adj_open AS DOUBLE),
        CAST(adj_high AS DOUBLE),
        CAST(adj_low AS DOUBLE),
        CAST(adj_close AS DOUBLE),
        CAST(adj_volume AS DOUBLE),
        CAST(dividend_cash AS DOUBLE),
        CAST(split_factor AS DOUBLE)
    FROM read_csv(
        ?,
        header = true,
        delim = ',',
        quote = '"',
        escape = '"',
        all_varchar = true
    )
    ON CONFLICT (security_id, date) DO UPDATE SET
        ticker = EXCLUDED.ticker,
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        adj_open = EXCLUDED.adj_open,
        adj_high = EXCLUDED.adj_high,
        adj_low = EXCLUDED.adj_low,
        adj_close = EXCLUDED.adj_close,
        adj_volume = EXCLUDED.adj_volume,
        dividend_cash = EXCLUDED.dividend_cash,
        split_factor = EXCLUDED.split_factor
"""


@dataclass(frozen=True, slots=True)
class _CachedSeries:
    bars: tuple[MarketBar, ...]
    dates: tuple[date, ...]


class DuckDbMarketStore:
    """Dense daily market store keyed by traded security and trading date.

    Legal-company identity is stored separately in ``company_security_map``.
    This allows one continuous security history to be referenced by multiple SEC
    CIKs without duplicating bars or pretending those legal entities are the
    same company.

    Databases created before the security-identity migration may still contain
    the legacy ``market_daily(company_id, ...)`` table. It is never written by
    this class; ``migrate_legacy_company_bars`` can copy verified rows lazily
    after a company/security mapping has been established.

    Historical experiments can enable a bounded read-through cache. The first
    lookup for a security loads its ordered series once; subsequent PIT slices
    use binary search in memory instead of reopening DuckDB for every candidate.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_cache_size = 0
        self._series_cache: OrderedDict[str, _CachedSeries] = OrderedDict()
        self._mapping_cache: dict[str, tuple[tuple[str, date, date | None], ...]] = {}
        self._series_cache_hits = 0
        self._series_cache_misses = 0
        self._mapping_cache_hits = 0
        self._mapping_cache_misses = 0
        self._initialize()

    def _connect(self):
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for market storage") from exc
        return duckdb.connect(str(self.database_path))

    def enable_read_cache(self, *, max_series: int = 8) -> None:
        """Enable bounded in-memory historical reads for repeated PIT queries."""

        if max_series <= 0:
            raise ValueError("max_series must be > 0")
        self._read_cache_size = max_series
        self.clear_read_cache()

    def disable_read_cache(self) -> None:
        self._read_cache_size = 0
        self.clear_read_cache()

    def clear_read_cache(self) -> None:
        self._series_cache.clear()
        self._mapping_cache.clear()
        self._series_cache_hits = 0
        self._series_cache_misses = 0
        self._mapping_cache_hits = 0
        self._mapping_cache_misses = 0

    def read_cache_stats(self) -> dict[str, int]:
        return {
            "enabled": int(self._read_cache_size > 0),
            "max_series": self._read_cache_size,
            "cached_series": len(self._series_cache),
            "cached_companies": len(self._mapping_cache),
            "series_hits": self._series_cache_hits,
            "series_misses": self._series_cache_misses,
            "mapping_hits": self._mapping_cache_hits,
            "mapping_misses": self._mapping_cache_misses,
        }

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS security_master (
                    security_id VARCHAR PRIMARY KEY,
                    ticker VARCHAR NOT NULL,
                    exchange_code VARCHAR,
                    history_start DATE NOT NULL,
                    history_end DATE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS company_security_map (
                    company_id VARCHAR NOT NULL,
                    security_id VARCHAR NOT NULL,
                    ticker VARCHAR NOT NULL,
                    valid_from DATE NOT NULL,
                    valid_to DATE,
                    PRIMARY KEY (company_id, security_id, valid_from)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS security_market_daily (
                    security_id VARCHAR NOT NULL,
                    ticker VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    open DOUBLE NOT NULL,
                    high DOUBLE NOT NULL,
                    low DOUBLE NOT NULL,
                    close DOUBLE NOT NULL,
                    volume DOUBLE NOT NULL,
                    adj_open DOUBLE NOT NULL,
                    adj_high DOUBLE NOT NULL,
                    adj_low DOUBLE NOT NULL,
                    adj_close DOUBLE NOT NULL,
                    adj_volume DOUBLE NOT NULL,
                    dividend_cash DOUBLE NOT NULL,
                    split_factor DOUBLE NOT NULL,
                    PRIMARY KEY (security_id, date)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_security_market_ticker_date "
                "ON security_market_daily(ticker, date)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_company_security_company "
                "ON company_security_map(company_id, valid_from)"
            )

    def register_mapping(self, mapping: SecurityMapping) -> None:
        """Persist one verified company→security relation and security master row."""

        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT ticker, history_start
                FROM security_master
                WHERE security_id = ?
                """,
                [mapping.security_id],
            ).fetchone()
            if existing is not None and (
                existing[0] != mapping.ticker or existing[1] != mapping.valid_from
            ):
                raise ValueError(
                    f"security_id {mapping.security_id} changed provider identity"
                )

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO security_master (
                        security_id, ticker, exchange_code, history_start, history_end
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        mapping.security_id,
                        mapping.ticker,
                        mapping.exchange_code,
                        mapping.valid_from,
                        mapping.valid_to,
                    ],
                )
            else:
                connection.execute(
                    """
                    UPDATE security_master
                    SET exchange_code = ?, history_end = ?
                    WHERE security_id = ?
                    """,
                    [mapping.exchange_code, mapping.valid_to, mapping.security_id],
                )

            connection.execute(
                """
                INSERT INTO company_security_map (
                    company_id, security_id, ticker, valid_from, valid_to
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (company_id, security_id, valid_from) DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    valid_to = EXCLUDED.valid_to
                """,
                [
                    mapping.company_id,
                    mapping.security_id,
                    mapping.ticker,
                    mapping.valid_from,
                    mapping.valid_to,
                ],
            )
        self._mapping_cache.pop(mapping.company_id, None)

    def migrate_legacy_company_bars(self, mapping: SecurityMapping) -> int:
        """Copy verified legacy company-keyed bars into security-keyed storage.

        The legacy table is consulted only after the resolver has produced an
        explicit mapping. This prevents old company attribution from becoming
        authority for the new security identity layer.
        """

        with self._connect() as connection:
            legacy_exists = connection.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_name = 'market_daily'
                """
            ).fetchone()
            if legacy_exists is None or int(legacy_exists[0]) == 0:
                return 0

            before = connection.execute(
                "SELECT COUNT(*) FROM security_market_daily WHERE security_id = ?",
                [mapping.security_id],
            ).fetchone()
            before_count = int(before[0]) if before is not None else 0
            end_day = mapping.valid_to or date.max
            connection.execute(
                """
                INSERT INTO security_market_daily (
                    security_id, ticker, date,
                    open, high, low, close, volume,
                    adj_open, adj_high, adj_low, adj_close, adj_volume,
                    dividend_cash, split_factor
                )
                SELECT
                    ?, ticker, date,
                    open, high, low, close, volume,
                    adj_open, adj_high, adj_low, adj_close, adj_volume,
                    dividend_cash, split_factor
                FROM market_daily
                WHERE company_id = ? AND ticker = ? AND date BETWEEN ? AND ?
                ON CONFLICT (security_id, date) DO NOTHING
                """,
                [
                    mapping.security_id,
                    mapping.company_id,
                    mapping.ticker,
                    mapping.valid_from,
                    end_day,
                ],
            )
            after = connection.execute(
                "SELECT COUNT(*) FROM security_market_daily WHERE security_id = ?",
                [mapping.security_id],
            ).fetchone()
            after_count = int(after[0]) if after is not None else before_count
        if after_count != before_count:
            self._series_cache.pop(mapping.security_id, None)
        return after_count - before_count

    def put_many(self, bars: tuple[MarketBar, ...] | list[MarketBar]) -> None:
        """Upsert market bars through DuckDB's vectorized CSV reader."""

        if not bars:
            return

        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                suffix=".csv",
                prefix="stock-traiding-market-",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(_MARKET_COLUMNS)
                for bar in bars:
                    writer.writerow(
                        (
                            bar.security_id,
                            bar.ticker,
                            bar.date.isoformat(),
                            str(bar.open),
                            str(bar.high),
                            str(bar.low),
                            str(bar.close),
                            str(bar.volume),
                            str(bar.adj_open),
                            str(bar.adj_high),
                            str(bar.adj_low),
                            str(bar.adj_close),
                            str(bar.adj_volume),
                            str(bar.dividend_cash),
                            str(bar.split_factor),
                        )
                    )

            with self._connect() as connection:
                connection.execute(_BULK_UPSERT_SQL, [str(temporary_path)])
            for security_id in {bar.security_id for bar in bars}:
                self._series_cache.pop(security_id, None)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def security_for_company(self, company_id: str, day: date) -> str | None:
        if self._read_cache_size > 0:
            rows = self._company_mappings(company_id)
            unique = {
                security_id
                for security_id, valid_from, valid_to in rows
                if valid_from <= day and (valid_to is None or valid_to >= day)
            }
        else:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT DISTINCT security_id
                    FROM company_security_map
                    WHERE company_id = ?
                      AND valid_from <= ?
                      AND (valid_to IS NULL OR valid_to >= ?)
                    """,
                    [company_id, day, day],
                ).fetchall()
            unique = {row[0] for row in rows}
        if len(unique) > 1:
            raise ValueError(f"multiple active securities for {company_id} on {day}")
        return next(iter(unique), None)

    def date_bounds(self, security_id: str, ticker: str) -> tuple[date, date] | None:
        """Return stored first/last trading dates for one verified security."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT MIN(date), MAX(date)
                FROM security_market_daily
                WHERE security_id = ? AND ticker = ?
                """,
                [security_id, ticker],
            ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return row[0], row[1]

    def count_bars(
        self,
        security_id: str,
        ticker: str,
        start_day: date,
        end_day: date,
    ) -> int:
        if end_day < start_day:
            raise ValueError("end_day must be >= start_day")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM security_market_daily
                WHERE security_id = ? AND ticker = ? AND date BETWEEN ? AND ?
                """,
                [security_id, ticker, start_day, end_day],
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def next_bar_after(self, security_id: str, day: date) -> MarketBar | None:
        series = self._series_for_read(security_id)
        if series is not None:
            index = bisect_right(series.dates, day)
            return series.bars[index] if index < len(series.bars) else None

        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM security_market_daily
                WHERE security_id = ? AND date > ?
                ORDER BY date
                LIMIT 1
                """,
                [security_id, day],
            )
            row = _one_dict(cursor)
        return _market_bar(row) if row is not None else None

    def bar_on(self, security_id: str, day: date) -> MarketBar | None:
        series = self._series_for_read(security_id)
        if series is not None:
            index = bisect_left(series.dates, day)
            if index < len(series.dates) and series.dates[index] == day:
                return series.bars[index]
            return None

        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM security_market_daily WHERE security_id = ? AND date = ?",
                [security_id, day],
            )
            row = _one_dict(cursor)
        return _market_bar(row) if row is not None else None

    def bars_before(self, security_id: str, day: date, limit: int) -> list[MarketBar]:
        """Return up to `limit` completed bars strictly before `day`, oldest first."""

        if limit <= 0:
            raise ValueError("limit must be > 0")
        series = self._series_for_read(security_id)
        if series is not None:
            end = bisect_left(series.dates, day)
            start = max(0, end - limit)
            return list(series.bars[start:end])

        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM (
                    SELECT * FROM security_market_daily
                    WHERE security_id = ? AND date < ?
                    ORDER BY date DESC
                    LIMIT ?
                )
                ORDER BY date
                """,
                [security_id, day, limit],
            )
            rows = _all_dicts(cursor)
        return [_market_bar(row) for row in rows]

    def bars_from(self, security_id: str, start_day: date, limit: int) -> list[MarketBar]:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        series = self._series_for_read(security_id)
        if series is not None:
            start = bisect_left(series.dates, start_day)
            return list(series.bars[start : start + limit])

        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM security_market_daily
                WHERE security_id = ? AND date >= ?
                ORDER BY date
                LIMIT ?
                """,
                [security_id, start_day, limit],
            )
            rows = _all_dicts(cursor)
        return [_market_bar(row) for row in rows]

    def export_parquet(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        escaped = str(path).replace("'", "''")
        with self._connect() as connection:
            connection.execute(
                f"COPY (SELECT * FROM security_market_daily ORDER BY security_id, date) "
                f"TO '{escaped}' (FORMAT PARQUET)"
            )
        return path

    def _company_mappings(self, company_id: str) -> tuple[tuple[str, date, date | None], ...]:
        cached = self._mapping_cache.get(company_id)
        if cached is not None:
            self._mapping_cache_hits += 1
            return cached

        self._mapping_cache_misses += 1
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT security_id, valid_from, valid_to
                FROM company_security_map
                WHERE company_id = ?
                ORDER BY valid_from, security_id
                """,
                [company_id],
            ).fetchall()
        result = tuple((str(row[0]), row[1], row[2]) for row in rows)
        self._mapping_cache[company_id] = result
        return result

    def _series_for_read(self, security_id: str) -> _CachedSeries | None:
        if self._read_cache_size <= 0:
            return None

        cached = self._series_cache.get(security_id)
        if cached is not None:
            self._series_cache_hits += 1
            self._series_cache.move_to_end(security_id)
            return cached

        self._series_cache_misses += 1
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM security_market_daily
                WHERE security_id = ?
                ORDER BY date
                """,
                [security_id],
            )
            rows = _all_dicts(cursor)
        bars = tuple(_market_bar(row) for row in rows)
        cached = _CachedSeries(
            bars=bars,
            dates=tuple(bar.date for bar in bars),
        )
        self._series_cache[security_id] = cached
        self._series_cache.move_to_end(security_id)
        while len(self._series_cache) > self._read_cache_size:
            self._series_cache.popitem(last=False)
        return cached


def _market_bar(row: dict) -> MarketBar:
    return MarketBar(
        security_id=row["security_id"],
        ticker=row["ticker"],
        date=row["date"],
        open=Decimal(str(row["open"])),
        high=Decimal(str(row["high"])),
        low=Decimal(str(row["low"])),
        close=Decimal(str(row["close"])),
        volume=Decimal(str(row["volume"])),
        adj_open=Decimal(str(row["adj_open"])),
        adj_high=Decimal(str(row["adj_high"])),
        adj_low=Decimal(str(row["adj_low"])),
        adj_close=Decimal(str(row["adj_close"])),
        adj_volume=Decimal(str(row["adj_volume"])),
        dividend_cash=Decimal(str(row["dividend_cash"])),
        split_factor=Decimal(str(row["split_factor"])),
    )


def _one_dict(cursor) -> dict | None:
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    return dict(zip(columns, row, strict=True))


def _all_dicts(cursor) -> list[dict]:
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
