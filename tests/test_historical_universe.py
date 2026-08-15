import json
from datetime import date

import duckdb

from stock_trading.entities import company_id_from_sec_cik
from stock_trading.experiments import historical_universe as module
from stock_trading.experiments.historical_universe import (
    build_historical_universe,
    load_historical_universe_company_ids,
)
from stock_trading.experiments.sec_snapshot import SecUniverseSnapshot
from stock_trading.market import IssuerObservation


def _snapshot_and_observations():
    snapshot = SecUniverseSnapshot(
        schema_version=1,
        start_year=2012,
        start_quarter=1,
        end_year=2026,
        end_quarter=2,
        issuer_observations=4,
        unique_tickers=4,
        sec_companies=3,
        source_artifacts=(),
    )
    observations = (
        IssuerObservation("0000000001", "Alpha", "AAA", date(2012, 1, 3)),
        IssuerObservation("0000000001", "Alpha Renamed", "AAB", date(2018, 2, 1)),
        IssuerObservation("0000000002", "Beta", "BBB", date(2014, 1, 3)),
        IssuerObservation("0000000003", "Gamma", "CCC", date(2020, 1, 3)),
    )
    return snapshot, observations


def test_historical_universe_is_deterministic_and_keeps_ticker_history(
    tmp_path, monkeypatch
) -> None:
    snapshot, observations = _snapshot_and_observations()
    monkeypatch.setattr(
        module,
        "load_sec_universe_snapshot",
        lambda data_root: (snapshot, observations),
    )

    first = build_historical_universe(
        tmp_path,
        max_companies=3,
        seed="fixed-seed",
    )
    first_payload = json.loads(first.output_path.read_text(encoding="utf-8"))
    second_path = tmp_path / "manifests" / "second.json"
    second = build_historical_universe(
        tmp_path,
        max_companies=3,
        seed="fixed-seed",
        output_path=second_path,
    )
    second_payload = json.loads(second.output_path.read_text(encoding="utf-8"))

    assert first_payload["companies"] == second_payload["companies"]
    assert first_payload["selection_uses_market_outcomes"] is False
    alpha = next(
        row
        for row in first_payload["companies"]
        if row["company_id"] == company_id_from_sec_cik("0000000001")
    )
    assert alpha["tickers"] == ["AAA", "AAB"]
    assert alpha["first_observed_date"] == "2012-01-03"
    assert load_historical_universe_company_ids(first.output_path) == tuple(
        row["company_id"] for row in first_payload["companies"]
    )


def test_historical_universe_can_exclude_existing_market_companies(
    tmp_path, monkeypatch
) -> None:
    snapshot, observations = _snapshot_and_observations()
    monkeypatch.setattr(
        module,
        "load_sec_universe_snapshot",
        lambda data_root: (snapshot, observations),
    )
    market_db = tmp_path / "market.duckdb"
    excluded = company_id_from_sec_cik("0000000001")
    with duckdb.connect(str(market_db)) as connection:
        connection.execute("CREATE TABLE company_security_map(company_id VARCHAR)")
        connection.execute("INSERT INTO company_security_map VALUES (?)", [excluded])

    result = build_historical_universe(
        tmp_path,
        max_companies=2,
        seed="fixed-seed",
        exclude_market_db=market_db,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    selected = {row["company_id"] for row in payload["companies"]}

    assert excluded not in selected
    assert result.excluded_existing_companies == 1
    assert result.excluded_manifest_companies == 0
    assert result.excluded_total_companies == 1
    assert len(selected) == 2


def test_historical_universe_excludes_all_prior_manifest_companies_even_if_unmapped(
    tmp_path, monkeypatch
) -> None:
    snapshot, observations = _snapshot_and_observations()
    monkeypatch.setattr(
        module,
        "load_sec_universe_snapshot",
        lambda data_root: (snapshot, observations),
    )

    alpha = company_id_from_sec_cik("0000000001")
    beta = company_id_from_sec_cik("0000000002")
    gamma = company_id_from_sec_cik("0000000003")

    market_db = tmp_path / "market.duckdb"
    with duckdb.connect(str(market_db)) as connection:
        connection.execute("CREATE TABLE company_security_map(company_id VARCHAR)")
        connection.execute("INSERT INTO company_security_map VALUES (?)", [alpha])

    prior_manifest = tmp_path / "prior_holdout.json"
    prior_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "companies": [
                    {"company_id": alpha},
                    {"company_id": beta},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_historical_universe(
        tmp_path,
        max_companies=1,
        seed="second-holdout",
        exclude_market_db=market_db,
        exclude_universe_manifests=(prior_manifest,),
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    selected = {row["company_id"] for row in payload["companies"]}

    assert selected == {gamma}
    assert alpha not in selected
    assert beta not in selected
    assert result.excluded_existing_companies == 1
    assert result.excluded_manifest_companies == 2
    assert result.excluded_total_companies == 2
    assert result.candidate_companies == 1
    assert payload["exclude_universe_manifests"] == [str(prior_manifest)]
