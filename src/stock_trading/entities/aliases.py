from dataclasses import dataclass
from pathlib import Path

from stock_trading.core import Source


@dataclass(frozen=True, slots=True)
class ExternalEntityAlias:
    source: Source
    external_id: str
    company_id: str
    display_name: str | None
    resolution_basis: str


class DuckDbExternalEntityAliases:
    """Verified source identifier -> canonical company mappings.

    The identity invariant is `(source, external_id) -> company_id`. Display
    names and resolution notes are provenance metadata and may vary harmlessly
    across later source observations.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for external entity aliases") from exc
        return duckdb.connect(str(self.database_path))

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS external_entity_aliases (
                    source VARCHAR NOT NULL,
                    external_id VARCHAR NOT NULL,
                    company_id VARCHAR NOT NULL,
                    display_name VARCHAR,
                    resolution_basis VARCHAR NOT NULL,
                    PRIMARY KEY (source, external_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_external_alias_company
                ON external_entity_aliases(company_id)
                """
            )

    def add(self, alias: ExternalEntityAlias) -> None:
        normalized = _normalize_alias(alias)
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT company_id FROM external_entity_aliases
                WHERE source = ? AND external_id = ?
                """,
                [normalized.source.value, normalized.external_id],
            ).fetchone()
            if existing is not None:
                if existing[0] != normalized.company_id:
                    raise ValueError(
                        f"external identity {normalized.source.value}:{normalized.external_id} "
                        "is already mapped to a different canonical company"
                    )
                return

            connection.execute(
                """
                INSERT INTO external_entity_aliases
                (source, external_id, company_id, display_name, resolution_basis)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    normalized.source.value,
                    normalized.external_id,
                    normalized.company_id,
                    normalized.display_name,
                    normalized.resolution_basis,
                ],
            )

    def resolve(self, source: Source, external_id: str) -> str | None:
        normalized_id = _normalize_external_id(source, external_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT company_id FROM external_entity_aliases
                WHERE source = ? AND external_id = ?
                """,
                [source.value, normalized_id],
            ).fetchone()
        return row[0] if row is not None else None

    def aliases_for(self, company_id: str) -> tuple[ExternalEntityAlias, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source, external_id, company_id, display_name, resolution_basis
                FROM external_entity_aliases
                WHERE company_id = ?
                ORDER BY source, external_id
                """,
                [company_id],
            ).fetchall()
        return tuple(
            ExternalEntityAlias(
                source=Source(row[0]),
                external_id=row[1],
                company_id=row[2],
                display_name=row[3],
                resolution_basis=row[4],
            )
            for row in rows
        )


def _normalize_alias(alias: ExternalEntityAlias) -> ExternalEntityAlias:
    company_id = alias.company_id.strip()
    basis = alias.resolution_basis.strip()
    if not company_id:
        raise ValueError("company_id must not be empty")
    if not basis:
        raise ValueError("resolution_basis must not be empty")
    display_name = alias.display_name.strip() if alias.display_name else None
    return ExternalEntityAlias(
        source=alias.source,
        external_id=_normalize_external_id(alias.source, alias.external_id),
        company_id=company_id,
        display_name=display_name or None,
        resolution_basis=basis,
    )


def _normalize_external_id(source: Source, external_id: str) -> str:
    value = str(external_id).strip()
    if not value:
        raise ValueError("external_id must not be empty")
    if source is Source.USASPENDING:
        return value.upper()
    return value
