import csv
import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from stock_trading.core import Event, EventType, Source, as_utc

if TYPE_CHECKING:
    import duckdb


_EVENT_COLUMNS = (
    "event_id",
    "event_type",
    "company_id",
    "actor_id",
    "event_time",
    "public_time",
    "first_tradable_time",
    "source",
    "source_record_id",
    "payload_json",
    "semantic_json",
    "raw_artifact_id",
    "ingested_at",
    "event_index",
)
_NULL_SENTINEL = "__STOCK_TRAIDING_NULL__"
_BULK_INSERT_SQL = f"""
    INSERT INTO events (
        event_id, event_type, company_id, actor_id,
        event_time, public_time, first_tradable_time,
        source, source_record_id, payload_json, semantic_json,
        raw_artifact_id, ingested_at, event_index
    )
    SELECT
        event_id,
        event_type,
        company_id,
        actor_id,
        CAST(event_time AS TIMESTAMPTZ),
        CAST(public_time AS TIMESTAMPTZ),
        CAST(first_tradable_time AS TIMESTAMPTZ),
        source,
        source_record_id,
        CAST(payload_json AS JSON),
        CAST(semantic_json AS JSON),
        raw_artifact_id,
        CAST(ingested_at AS TIMESTAMPTZ),
        CAST(event_index AS INTEGER)
    FROM read_csv(
        ?,
        header = true,
        delim = ',',
        quote = '"',
        escape = '"',
        all_varchar = true,
        nullstr = '{_NULL_SENTINEL}'
    )
    ON CONFLICT (event_id) DO NOTHING
"""


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
                    ingested_at TIMESTAMPTZ NOT NULL,
                    event_index INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info('events')").fetchall()
            }
            if "event_index" not in columns:
                connection.execute(
                    "ALTER TABLE events ADD COLUMN event_index INTEGER DEFAULT 0"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_company_public ON events(company_id, public_time)"
            )

    def put(self, event: Event) -> None:
        self.put_many((event,))

    def put_many(self, events: tuple[Event, ...] | list[Event]) -> None:
        """Insert events through DuckDB's vectorized CSV reader in one statement.

        DuckDB explicitly warns against ``executemany`` for large ingestion. A
        temporary CSV keeps this path dependency-light while allowing DuckDB to
        parse and insert the quarter as a bulk relation. The target primary key
        still makes reruns idempotent.
        """

        if not events:
            return

        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                suffix=".csv",
                prefix="stock-traiding-events-",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(_EVENT_COLUMNS)
                for event in events:
                    writer.writerow(
                        (
                            event.event_id,
                            event.event_type.value,
                            _csv_value(event.company_id),
                            _csv_value(event.actor_id),
                            event.event_time.isoformat(),
                            event.public_time.isoformat(),
                            _csv_value(event.first_tradable_time),
                            event.source.value,
                            event.source_record_id,
                            event.payload.model_dump_json(),
                            _csv_value(
                                event.semantic.model_dump_json()
                                if event.semantic is not None
                                else None
                            ),
                            event.raw_artifact_id,
                            event.ingested_at.isoformat(),
                            event.event_index,
                        )
                    )

            with self._connect() as connection:
                connection.execute(_BULK_INSERT_SQL, [str(temporary_path)])
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

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
            return _all_dicts(cursor)

    def all_events(
        self,
        *,
        company_id: str | None = None,
        event_types: tuple[EventType, ...] | None = None,
    ) -> tuple[Event, ...]:
        clauses: list[str] = []
        params: list[object] = []
        if company_id is not None:
            clauses.append("company_id = ?")
            params.append(company_id)
        if event_types:
            placeholders = ", ".join("?" for _ in event_types)
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(event_type.value for event_type in event_types)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            cursor = connection.execute(
                f"SELECT * FROM events {where} ORDER BY public_time, event_id",
                params,
            )
            rows = _all_dicts(cursor)
        return tuple(_event_from_row(row) for row in rows)

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


def _csv_value(value: object | None) -> object:
    if value is None:
        return _NULL_SENTINEL
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _event_from_row(row: dict) -> Event:
    payload = row["payload_json"]
    semantic = row["semantic_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(semantic, str):
        semantic = json.loads(semantic)

    return Event(
        event_id=row["event_id"],
        event_type=EventType(row["event_type"]),
        event_index=int(row.get("event_index") or 0),
        company_id=row["company_id"],
        actor_id=row["actor_id"],
        event_time=row["event_time"],
        public_time=row["public_time"],
        first_tradable_time=row["first_tradable_time"],
        source=Source(row["source"]),
        source_record_id=row["source_record_id"],
        payload=payload,
        semantic=semantic,
        raw_artifact_id=row["raw_artifact_id"],
        ingested_at=row["ingested_at"],
    )


def _all_dicts(cursor) -> list[dict]:
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
