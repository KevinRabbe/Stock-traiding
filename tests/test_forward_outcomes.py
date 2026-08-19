import json
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from stock_trading.live.forward_outcomes import refresh_forward_outcome_scorecard
from stock_trading.market import DuckDbMarketStore, MarketBar


UTC = timezone.utc


def _bar(
    security_id: str,
    ticker: str,
    day: date,
    *,
    open_price: str,
    high_price: str,
    low_price: str,
    close_price: str,
) -> MarketBar:
    return MarketBar(
        security_id=security_id,
        ticker=ticker,
        date=day,
        open=Decimal(open_price),
        high=Decimal(high_price),
        low=Decimal(low_price),
        close=Decimal(close_price),
        volume=Decimal("1000"),
        adj_open=Decimal(open_price),
        adj_high=Decimal(high_price),
        adj_low=Decimal(low_price),
        adj_close=Decimal(close_price),
        adj_volume=Decimal("1000"),
        dividend_cash=Decimal("0"),
        split_factor=Decimal("1"),
    )


def _decision(candidate_id: str, *, emitted: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "company_id": "cmp_test",
        "security_id": "sec_test",
        "execution_date": "2026-08-19",
        "chosen_horizon": 5 if emitted else None,
        "final_percentile": 0.95 if emitted else 0.0,
        "rank_threshold": 0.9,
        "emitted": emitted,
        "rejection_reason": "emitted" if emitted else "no_eligible_horizon",
        "horizons": [
            {
                "horizon_sessions": horizon,
                "expected_return": 0.01,
                "expected_alpha": 0.005,
                "expected_downside": 0.02,
                "probability_positive": 0.6,
                "raw_profit_score": 0.1,
                "profit_percentile": 0.7,
                "alpha_percentile": 0.6,
                "combined_signal": 0.65,
                "eligible": emitted and horizon == 5,
                "eligibility_reasons": [],
                "required_feature_count": 10,
                "missing_feature_count": 0,
                "missing_feature_names": [],
            }
            for horizon in (5, 20, 60)
        ],
    }


def _write_diagnostic(runtime_dir, batch_id: str, candidate_id: str) -> None:
    root = runtime_dir / "decision_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "batch_id": batch_id,
        "as_of": "2026-08-18T18:00:00+00:00",
        "target_execution_date": "2026-08-19",
        "strategies": [
            {
                "strategy_id": "champion",
                "candidate_count": 1,
                "emitted_opportunity_count": 0,
                "decisions": [_decision(candidate_id, emitted=False)],
            },
            {
                "strategy_id": "shadow",
                "candidate_count": 1,
                "emitted_opportunity_count": 1,
                "decisions": [_decision(candidate_id, emitted=True)],
            },
        ],
    }
    (root / f"{batch_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _seed_market(market_db) -> None:
    store = DuckDbMarketStore(market_db)
    sessions = (
        date(2026, 8, 19),
        date(2026, 8, 20),
        date(2026, 8, 21),
        date(2026, 8, 24),
        date(2026, 8, 25),
    )
    stock_closes = ("11", "12", "13", "14", "15")
    benchmark_closes = ("101", "102", "103", "104", "105")
    bars = []
    for index, day in enumerate(sessions):
        bars.append(
            _bar(
                "sec_test",
                "TEST",
                day,
                open_price="10" if index == 0 else stock_closes[index - 1],
                high_price="16" if index == 4 else stock_closes[index],
                low_price="9" if index == 0 else stock_closes[index - 1],
                close_price=stock_closes[index],
            )
        )
        bars.append(
            _bar(
                "sec_spy",
                "SPY",
                day,
                open_price="100" if index == 0 else benchmark_closes[index - 1],
                high_price=benchmark_closes[index],
                low_price="99" if index == 0 else benchmark_closes[index - 1],
                close_price=benchmark_closes[index],
            )
        )
    store.put_many(bars)


def test_forward_scorecard_matures_rejected_and_emitted_decisions(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    market_db = tmp_path / "market.duckdb"
    runtime_dir.mkdir()
    (runtime_dir / "paper_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "market_db": str(market_db),
                "benchmark_security_id": "sec_spy",
                "paper_ledger": str(runtime_dir / "paper_ledger.json"),
            }
        ),
        encoding="utf-8",
    )
    _seed_market(market_db)
    candidate_id = "opportunity:cmp_test:2026-08-19"
    _write_diagnostic(runtime_dir, "batch_one", candidate_id)

    result = refresh_forward_outcome_scorecard(
        data_root=tmp_path / "data",
        runtime_dir=runtime_dir,
        as_of=datetime(2026, 8, 25, 21, 0, tzinfo=UTC),
    )

    assert result["candidate_decision_instance_count"] == 1
    assert result["strategy_decision_count"] == 2
    assert result["emitted_strategy_decision_count"] == 1
    assert result["rejected_strategy_decision_count"] == 1
    assert result["matured_by_horizon"] == {"5": 1, "20": 0, "60": 0}
    assert result["pending_horizon_label_count"] == 2

    scorecard = json.loads((runtime_dir / "forward_scorecard.json").read_text(encoding="utf-8"))
    label = scorecard["observations"][0]["realized_labels"]["5"]
    assert label["stock_return"] == pytest.approx(0.5)
    assert label["benchmark_return"] == pytest.approx(0.05)
    assert label["alpha"] == pytest.approx(0.45)
    assert label["max_favorable_excursion"] == pytest.approx(0.6)
    assert label["max_adverse_excursion"] == pytest.approx(-0.1)


def test_forward_scorecard_keeps_same_candidate_id_from_two_batches_distinct(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    market_db = tmp_path / "market.duckdb"
    runtime_dir.mkdir()
    (runtime_dir / "paper_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "market_db": str(market_db),
                "benchmark_security_id": "sec_spy",
                "paper_ledger": str(runtime_dir / "paper_ledger.json"),
            }
        ),
        encoding="utf-8",
    )
    _seed_market(market_db)
    candidate_id = "opportunity:cmp_test:2026-08-19"
    _write_diagnostic(runtime_dir, "batch_one", candidate_id)
    _write_diagnostic(runtime_dir, "batch_two", candidate_id)

    result = refresh_forward_outcome_scorecard(
        data_root=tmp_path / "data",
        runtime_dir=runtime_dir,
        as_of=datetime(2026, 8, 25, 21, 0, tzinfo=UTC),
    )

    assert result["diagnostic_batch_count"] == 2
    assert result["candidate_decision_instance_count"] == 2
    assert result["matured_by_horizon"]["5"] == 2
    scorecard = json.loads((runtime_dir / "forward_scorecard.json").read_text(encoding="utf-8"))
    assert {item["observation_id"] for item in scorecard["observations"]} == {
        f"batch_one:{candidate_id}",
        f"batch_two:{candidate_id}",
    }
