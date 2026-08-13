import csv
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import MarketBar


_MARKET_COLUMNS = (
    "company_id",
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
    INSERT INTO market_daily (
        company_id, ticker, date,
        open, high, low, close, volume,
        adj_open, adj_high, adj_low, adj_close, adj_volume,
        dividend_cash, split_factor
    )
    SELECT
        company_id,
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
    ON CONFLICT (company_id, date) DO UPDATE SET
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


class DuckDbMarketStore:
    """Dense daily market store keyed by canonical company and trading date."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for market storage") from exc
        return duckdb.connect(str(self.database_path))

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS market_daily (
                    company_id VARCHAR NOT NULL,
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
                    PRIMARY KEY (company_id, date)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_ticker_date ON market_daily(ticker, date)"
            )

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
                            bar.company_id,
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
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def date_bounds(self, company_id: str, ticker: str) -> tuple[date, date] | None:
        """Return stored first/last trading dates for one verified security mapping."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT MIN(date), MAX(date)
                FROM market_daily
                WHERE company_id = ? AND ticker = ?
                """,
                [company_id, ticker],
            ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return row[0], row[1]

    def count_bars(
        self,
        company_id: str,
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
                FROM market_daily
                WHERE company_id = ? AND ticker = ? AND date BETWEEN ? AND ?
                """,
                [company_id, ticker, start_day, end_day],
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def next_bar_after(self, company_id: str, day: date) -> MarketBar | None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM market_daily
                WHERE company_id = ? AND date > ?
                ORDER BY date
                LIMIT 1
                """,
                [company_id, day],
            )
            row = _one_dict(cursor)
        return _market_bar(row) if row is not None else None

    def bar_on(self, company_id: str, day: date) -> MarketBar | None:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM market_daily WHERE company_id = ? AND date = ?",
                [company_id, day],
            )
            row = _one_dict(cursor)
        return _market_bar(row) if row is not None else None

    def bars_before(self, company_id: str, day: date, limit: int) -> list[MarketBar]:
        """Return up to `limit` completed bars strictly before `day`, oldest first."""

        if limit <= 0:
            raise ValueError("limit must be > 0")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM (
                    SELECT * FROM market_daily
                    WHERE company_id = ? AND date < ?
                    ORDER BY date DESC
                    LIMIT ?
                )
                ORDER BY date
                """,
                [company_id, day, limit],
            )
            rows = _all_dicts(cursor)
        return [_market_bar(row) for row in rows]

    def bars_from(self, company_id: str, start_day: date, limit: int) -> list[MarketBar]:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM market_daily
                WHERE company_id = ? AND date >= ?
                ORDER BY date
                LIMIT ?
                """,
                [company_id, start_day, limit],
            )
            rows = _all_dicts(cursor)
        return [_market_bar(row) for row in rows]

    def export_parquet(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        escaped = str(path).replace("'", "''")
        with self._connect() as connection:
            connection.execute(
                f"COPY (SELECT * FROM market_daily ORDER BY company_id, date) "
                f"TO '{escaped}' (FORMAT PARQUET)"
            )
        return path


def _market_bar(row: dict) -> MarketBar:
    return MarketBar(
        company_id=row["company_id"],
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
