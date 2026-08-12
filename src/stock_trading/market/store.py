from datetime import date
from decimal import Decimal
from pathlib import Path

from .models import MarketBar


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
        if not bars:
            return
        rows = [
            (
                bar.company_id,
                bar.ticker,
                bar.date,
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
                float(bar.adj_open),
                float(bar.adj_high),
                float(bar.adj_low),
                float(bar.adj_close),
                float(bar.adj_volume),
                float(bar.dividend_cash),
                float(bar.split_factor),
            )
            for bar in bars
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO market_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                """,
                rows,
            )

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
