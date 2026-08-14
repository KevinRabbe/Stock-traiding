import io
import json
import zipfile
from datetime import datetime, timezone

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.experiments.market_prepare import _parser as market_parser
from stock_trading.experiments.market_prepare import _select_observations
from stock_trading.experiments.sec_snapshot import (
    build_sec_universe_snapshot,
    load_sec_universe_snapshot,
)
from stock_trading.market import IssuerObservation
from stock_trading.storage import FileRawStore


def _quarter_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "SUBMISSION.tsv",
            "\t".join(
                [
                    "ACCESSION_NUMBER",
                    "DOCUMENT_TYPE",
                    "FILING_DATE",
                    "ISSUERCIK",
                    "ISSUERNAME",
                    "ISSUERTRADINGSYMBOL",
                    "AFF10B5ONE",
                ]
            )
            + "\n"
            + "\t".join(
                [
                    "0000000001-20-000001",
                    "4",
                    "2020-01-03",
                    "12345",
                    "Example Corp",
                    "EXM",
                    "0",
                ]
            )
            + "\n",
        )
        archive.writestr(
            "REPORTINGOWNER.tsv",
            "\t".join(
                [
                    "ACCESSION_NUMBER",
                    "RPTOWNERCIK",
                    "RPTOWNERNAME",
                    "RPTOWNER_RELATIONSHIP",
                    "RPTOWNER_TITLE",
                ]
            )
            + "\n",
        )
        archive.writestr(
            "NONDERIV_TRANS.tsv",
            "\t".join(
                [
                    "ACCESSION_NUMBER",
                    "NONDERIV_TRANS_SK",
                    "TRANS_DATE",
                    "SECURITY_TITLE",
                    "TRANS_CODE",
                    "TRANS_ACQUIRED_DISP_CD",
                    "TRANS_SHARES",
                    "TRANS_PRICEPERSHARE",
                    "SHRS_OWND_FOLWNG_TRANS",
                    "DIRECT_INDIRECT_OWNERSHIP",
                    "NATURE_OF_OWNERSHIP",
                ]
            )
            + "\n"
            + "\t".join(
                [
                    "0000000001-20-000001",
                    "1",
                    "2020-01-02",
                    "Common Stock",
                    "P",
                    "A",
                    "100",
                    "10",
                    "1000",
                    "D",
                    "",
                ]
            )
            + "\n",
        )
    return buffer.getvalue()


def test_sec_snapshot_builds_once_from_cached_quarters_and_round_trips(tmp_path) -> None:
    content = _quarter_zip()
    raw = RawRecord(
        source=Source.SEC_QUARTERLY,
        source_record_id="2020Q1",
        fetched_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        content_type="application/zip",
        content=content,
        sha256=content_sha256(content),
    )
    FileRawStore(tmp_path / "raw").put(raw)

    snapshot = build_sec_universe_snapshot(
        tmp_path,
        start_year=2020,
        start_quarter=1,
        end_year=2020,
        end_quarter=1,
    )
    restored, observations = load_sec_universe_snapshot(tmp_path)

    assert restored == snapshot
    assert snapshot.issuer_observations == 1
    assert snapshot.unique_tickers == 1
    assert snapshot.sec_companies == 1
    assert snapshot.source_artifacts[0]["sha256"] == raw.sha256
    assert observations == (
        IssuerObservation(
            sec_cik="0000012345",
            issuer_name="Example Corp",
            ticker="EXM",
            observed_date=datetime(2020, 1, 3).date(),
        ),
    )

    metadata = json.loads((tmp_path / "manifests" / "sec_snapshot.json").read_text())
    assert metadata["start_year"] == 2020
    assert metadata["end_quarter"] == 1


def test_market_only_cli_needs_no_sec_identity_and_selection_is_deterministic() -> None:
    args = market_parser().parse_args(
        ["--tickers", "aapl", "MSFT", "--market-end", "2026-08-13"]
    )
    assert args.tickers == ["aapl", "MSFT"]
    assert not hasattr(args, "sec_user_agent")

    observations = (
        IssuerObservation(
            sec_cik="0000000001",
            issuer_name="Apple Inc",
            ticker='"AAPL"',
            observed_date=datetime(2020, 1, 1).date(),
        ),
        IssuerObservation(
            sec_cik="0000000002",
            issuer_name="Microsoft Corp",
            ticker="MSFT",
            observed_date=datetime(2020, 1, 1).date(),
        ),
    )
    selected, count, requested, missing = _select_observations(
        observations,
        max_unique_tickers=None,
        requested_tickers=("AAPL", "NVDA"),
    )

    assert selected == (observations[0],)
    assert count == 1
    assert requested == ("AAPL", "NVDA")
    assert missing == ("NVDA",)
