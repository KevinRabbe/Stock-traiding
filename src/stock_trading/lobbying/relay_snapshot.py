from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.core import as_utc

from .client import LdaClient

RELAY_SCHEMA_VERSION = 1
DEFAULT_LOOKBACK_DAYS = 14


def build_relay_snapshot(
    *,
    client: LdaClient,
    as_of: datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict:
    """Fetch a deterministic rolling window from the official LDA API."""

    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    cutoff = as_utc(as_of or datetime.now(timezone.utc))
    posted_after = cutoff.date() - timedelta(days=lookback_days)
    posted_before = cutoff.date()

    filings_by_uuid: dict[str, dict] = {}
    page = 1
    pages_fetched = 0
    while True:
        raw = client.fetch_filings_page(
            posted_after=posted_after,
            posted_before=posted_before,
            page=page,
            page_size=25,
        )
        payload = json.loads(raw.content.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("LDA relay page must be a JSON object")
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("LDA relay page must contain a results list")
        pages_fetched += 1

        for filing in results:
            if not isinstance(filing, dict):
                raise ValueError("LDA relay filing must be a JSON object")
            filing_uuid = str(filing.get("filing_uuid") or "").strip()
            if not filing_uuid:
                raise ValueError("LDA relay filing is missing filing_uuid")
            previous = filings_by_uuid.get(filing_uuid)
            if previous is not None and previous != filing:
                raise ValueError(f"LDA relay filing changed within one snapshot: {filing_uuid}")
            filings_by_uuid[filing_uuid] = filing

        if not payload.get("next"):
            break
        page += 1

    filings = sorted(filings_by_uuid.values(), key=_filing_sort_key)
    return {
        "schema_version": RELAY_SCHEMA_VERSION,
        "generated_at": cutoff.isoformat(),
        "source_base_url": client.base_url,
        "posted_after": posted_after.isoformat(),
        "posted_before": posted_before.isoformat(),
        "pages_fetched": pages_fetched,
        "filing_count": len(filings),
        "filings": filings,
    }


def write_relay_snapshot(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _filing_sort_key(filing: dict) -> tuple[str, str]:
    return (
        str(filing.get("dt_posted") or ""),
        str(filing.get("filing_uuid") or ""),
    )


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a rolling official-LDA snapshot for the GitHub relay branch."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--as-of")
    args = parser.parse_args()

    # Relay publishing runs from GitHub Actions, whose egress can reach the official
    # endpoint. Disable relay fallback here so a broken source can never recursively
    # consume its own previously published snapshot.
    with LdaClient(relay_fallback=False) as client:
        payload = build_relay_snapshot(
            client=client,
            as_of=_parse_as_of(args.as_of),
            lookback_days=args.lookback_days,
        )
    write_relay_snapshot(args.output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "generated_at": payload["generated_at"],
                "posted_after": payload["posted_after"],
                "posted_before": payload["posted_before"],
                "pages_fetched": payload["pages_fetched"],
                "filing_count": payload["filing_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
