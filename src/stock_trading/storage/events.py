from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from stock_trading.core import Event, as_utc

if TYPE_CHECKING:
    import duckdb


class DuckDbEventStore:
    """Idempotent normalized sparse-event store backed by DuckDB."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for normalized event storage") from exc
        return duckdb.connect(str(self.database_path))

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id VARCHAR PRIMARY KEY,
                    event_type VARCHAR NOT NULL,
                    company_id VARCHAR,
                    actor_id VARCHAR,
                    event_time TIMESTAMPTZ NOT NULL,
                    public_time TIMESTAMPTZ NOT NULL,
                    first_tradable_time TIMESTAMPTZ,
                    source VARCHAR NOT NULL,
                    source_record_id VARCHAR NOT NULL,
                    payload_json JSON NOT NULL,
                    semantic_json JSON,
                    raw_artifact_id VARCHAR NOT NULL,
                    ingested_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_company_public ON events(company_id, public_time)"
            )

    def put(self, event: Event) -> None:
        self.put_many((event,))

    def put_many(self, events: tuple[Event, ...] | list[Event]) -> None:
        if not events:
            return
        rows = [
            (
                event.event_id,
                event.event_type.value,
                event.company_id,
                event.actor_id,
                event.event_time,
                event.public_time,
                event.first_tradable_time,
                event.source.value,
                event.source_record_id,
                event.payload.model_dump_json(),
                event.semantic.model_dump_json() if event.semantic is not None else None,
                event.raw_artifact_id,
                event.ingested_at,
            )
            for event in events
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (event_id) DO NOTHING
                """,
                rows,
            )

    def public_rows(self, company_id: str, decision_time: datetime) -> list[dict]:
        cutoff = as_utc(decision_time)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM events
                WHERE company_id = ? AND public_time <= ?
                ORDER BY public_time, event_id
                """,
                [company_id, cutoff],
            )
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def export_parquet(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        escaped = str(path).replace("'", "''")
        with self._connect() as connection:
            connection.execute(
                f"COPY (SELECT * FROM events ORDER BY public_time, event_id) "
                f"TO '{escaped}' (FORMAT PARQUET)"
            )
        return path
