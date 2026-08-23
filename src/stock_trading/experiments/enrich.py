import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from stock_trading.core import Event, Source
from stock_trading.entities import DuckDbExternalEntityAliases, ExternalEntityAlias
from stock_trading.extraction import FileSemanticCache, QwenSemanticExtractor
from stock_trading.extraction.qwen import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL
from stock_trading.lobbying import LdaClient, LdaFilingNormalizer
from stock_trading.market import normalize_company_name
from stock_trading.storage import DuckDbEventStore, FileRawStore


@dataclass(frozen=True, slots=True)
class LdaEnrichmentConfig:
    data_root: Path
    start_year: int = 2012
    end_year: int | None = None
    max_pages_per_year: int | None = None
    qwen_base_url: str = DEFAULT_QWEN_BASE_URL
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_extractor_version: str = "semantic-v1"


@dataclass(frozen=True, slots=True)
class LdaEnrichmentResult:
    pages_downloaded: int
    filings_seen: int
    events_stored: int
    mapped_events: int
    qwen_enriched_events: int
    unresolved_clients: int
    ambiguous_clients: int
    events_db: Path
    aliases_db: Path


def enrich_lda_and_qwen(
    config: LdaEnrichmentConfig,
    *,
    lda_client: LdaClient,
    extractor: QwenSemanticExtractor,
) -> LdaEnrichmentResult:
    if config.start_year < 1995:
        raise ValueError("start_year is implausibly early for LDA data")
    if config.max_pages_per_year is not None and config.max_pages_per_year <= 0:
        raise ValueError("max_pages_per_year must be > 0")

    data_root = config.data_root
    companies_path = data_root / "manifests" / "sec_companies.jsonl"
    if not companies_path.exists():
        raise ValueError(
            "SEC company manifest is missing; run the SEC/market preparation pass first"
        )

    name_index = load_unique_company_name_index(companies_path)
    raw_store = FileRawStore(data_root / "raw")
    events_db = data_root / "normalized" / "events.duckdb"
    aliases_db = data_root / "normalized" / "aliases.duckdb"
    event_store = DuckDbEventStore(events_db)
    alias_store = DuckDbExternalEntityAliases(aliases_db)
    normalizer = LdaFilingNormalizer()

    end_year = config.end_year if config.end_year is not None else _current_year()
    if end_year < config.start_year:
        raise ValueError("end_year must not precede start_year")

    pages_downloaded = 0
    filings_seen = 0
    events_stored = 0
    mapped_events = 0
    qwen_enriched_events = 0
    unresolved_names: set[tuple[int | None, str]] = set()
    ambiguous_names: set[tuple[int | None, str]] = set()

    for year in range(config.start_year, end_year + 1):
        page = 1
        while True:
            if config.max_pages_per_year is not None and page > config.max_pages_per_year:
                break

            raw = lda_client.fetch_filings_page(
                filing_year=year,
                page=page,
                page_size=25,
            )
            raw_store.put(raw)
            response = _json_payload(raw.content)
            results = response.get("results") if isinstance(response, dict) else None
            if not isinstance(results, list):
                raise ValueError("LDA page response has no results list")
            if not results:
                break

            pages_downloaded += 1
            filings_seen += len(results)
            company_ids_by_client_id: dict[int, str] = {}
            for filing in results:
                if not isinstance(filing, dict):
                    continue
                client = filing.get("client") or {}
                if not isinstance(client, dict):
                    continue
                client_id = _int_or_none(client.get("id"))
                client_name = str(client.get("name") or "").strip()
                if client_id is None or not client_name:
                    continue

                normalized_name = normalize_company_name(client_name)
                matches = name_index.get(normalized_name, ())
                if len(matches) == 1:
                    company_id = matches[0]
                    company_ids_by_client_id[client_id] = company_id
                    alias_store.add(
                        ExternalEntityAlias(
                            source=Source.LDA,
                            external_id=str(client_id),
                            company_id=company_id,
                            display_name=client_name,
                            resolution_basis="unique normalized SEC issuer-name match",
                        )
                    )
                elif len(matches) > 1:
                    ambiguous_names.add((client_id, client_name))
                else:
                    unresolved_names.add((client_id, client_name))

            events = normalizer.to_events(
                raw,
                company_ids_by_client_id=company_ids_by_client_id,
            )
            enriched: list[Event] = []
            for event in events:
                if event.company_id is None:
                    enriched.append(event)
                    continue

                mapped_events += 1
                semantic = extractor.extract(
                    normalizer.semantic_text(event),
                    context="US federal lobbying disclosure",
                )
                enriched.append(event.model_copy(update={"semantic": semantic}))
                qwen_enriched_events += 1

            event_store.put_many(enriched)
            events_stored += len(enriched)

            next_url = response.get("next") if isinstance(response, dict) else None
            if not next_url:
                break
            page += 1

    manifests = data_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    unresolved_path = manifests / "unresolved_lda_clients.jsonl"
    with unresolved_path.open("w", encoding="utf-8") as handle:
        for client_id, client_name in sorted(unresolved_names, key=lambda item: (item[1], item[0] or -1)):
            handle.write(
                json.dumps(
                    {
                        "client_id": client_id,
                        "client_name": client_name,
                        "reason": "no_unique_sec_name_match",
                    },
                    sort_keys=True,
                )
            )
            handle.write("\n")
        for client_id, client_name in sorted(ambiguous_names, key=lambda item: (item[1], item[0] or -1)):
            handle.write(
                json.dumps(
                    {
                        "client_id": client_id,
                        "client_name": client_name,
                        "reason": "ambiguous_sec_name_match",
                    },
                    sort_keys=True,
                )
            )
            handle.write("\n")

    result = LdaEnrichmentResult(
        pages_downloaded=pages_downloaded,
        filings_seen=filings_seen,
        events_stored=events_stored,
        mapped_events=mapped_events,
        qwen_enriched_events=qwen_enriched_events,
        unresolved_clients=len(unresolved_names),
        ambiguous_clients=len(ambiguous_names),
        events_db=events_db,
        aliases_db=aliases_db,
    )
    (manifests / "lda_qwen.json").write_text(
        json.dumps(
            {
                **_jsonable(asdict(result)),
                "start_year": config.start_year,
                "end_year": end_year,
                "qwen_base_url": config.qwen_base_url,
                "qwen_model": config.qwen_model,
                "qwen_extractor_version": config.qwen_extractor_version,
                "resolution_basis": "unique normalized SEC issuer-name match",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result


def load_unique_company_name_index(path: Path) -> dict[str, tuple[str, ...]]:
    matches: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            company_id = str(row["company_id"])
            for name in row.get("issuer_names", []):
                normalized = normalize_company_name(str(name))
                if normalized:
                    matches.setdefault(normalized, set()).add(company_id)
    return {
        name: tuple(sorted(company_ids))
        for name, company_ids in matches.items()
    }


def _json_payload(content: bytes | str) -> dict:
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _int_or_none(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _current_year() -> int:
    from datetime import date

    return date.today().year


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich historical LDA lobbying filings with local Qwen semantics."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--max-pages-per-year", type=int)
    parser.add_argument(
        "--qwen-base-url",
        default=os.environ.get("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    api_key = os.environ.get("LDA_API_TOKEN", "").strip() or None
    config = LdaEnrichmentConfig(
        data_root=args.data_root,
        start_year=args.start_year,
        end_year=args.end_year,
        max_pages_per_year=args.max_pages_per_year,
        qwen_base_url=args.qwen_base_url,
        qwen_model=args.qwen_model,
    )
    with LdaClient(api_key=api_key) as lda_client, QwenSemanticExtractor(
        cache=FileSemanticCache(args.data_root / "cache" / "semantic"),
        base_url=config.qwen_base_url,
        model=config.qwen_model,
        extractor_version=config.qwen_extractor_version,
    ) as extractor:
        result = enrich_lda_and_qwen(
            config,
            lda_client=lda_client,
            extractor=extractor,
        )
    print(json.dumps(_jsonable(asdict(result)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
