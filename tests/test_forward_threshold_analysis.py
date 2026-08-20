import json

import pytest

from stock_trading.live.analyze_forward_thresholds import (
    analyze_forward_rank_thresholds,
)


def _decision(
    strategy_id: str,
    *,
    chosen_horizon,
    final_percentile: float,
    rank_threshold: float = 0.95,
    emitted: bool = False,
):
    return {
        "strategy_id": strategy_id,
        "emitted": emitted,
        "rejection_reason": "emitted" if emitted else "below_final_rank_threshold",
        "chosen_horizon": chosen_horizon,
        "final_percentile": final_percentile,
        "rank_threshold": rank_threshold,
        "horizons": [],
    }


def _observation(observation_id: str, decisions: list[dict], labels: dict) -> dict:
    return {
        "observation_id": observation_id,
        "batch_id": observation_id.split(":", 1)[0],
        "candidate_id": observation_id.split(":", 1)[1],
        "company_id": "cmp_test",
        "security_id": "sec_test",
        "execution_date": "2026-08-20",
        "strategy_decisions": decisions,
        "realized_labels": labels,
        "matured_horizon_count": len(labels),
        "fully_matured": False,
    }


def _label(
    *,
    stock_return: float,
    alpha: float,
    mfe: float,
    mae: float,
) -> dict:
    return {
        "horizon_sessions": 5,
        "start_date": "2026-08-20",
        "end_date": "2026-08-26",
        "stock_return": stock_return,
        "benchmark_return": stock_return - alpha,
        "alpha": alpha,
        "max_favorable_excursion": mfe,
        "max_adverse_excursion": mae,
    }


def _write_scorecard(runtime_dir, observations: list[dict]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "forward_scorecard.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-08-27T21:00:00+00:00",
                "as_of": "2026-08-27T21:00:00+00:00",
                "last_completed_xnys_session": "2026-08-27",
                "summary": {},
                "market_sync": {},
                "observations": observations,
            }
        ),
        encoding="utf-8",
    )


def test_threshold_analysis_changes_only_final_rank_gate(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    observations = [
        _observation(
            "batch_a:candidate_a",
            [
                _decision(
                    "champion",
                    chosen_horizon=5,
                    final_percentile=0.96,
                    emitted=True,
                )
            ],
            {
                "5": _label(
                    stock_return=0.10,
                    alpha=0.08,
                    mfe=0.15,
                    mae=-0.04,
                )
            },
        ),
        _observation(
            "batch_b:candidate_b",
            [
                _decision(
                    "champion",
                    chosen_horizon=5,
                    final_percentile=0.75,
                )
            ],
            {
                "5": _label(
                    stock_return=0.04,
                    alpha=-0.01,
                    mfe=0.09,
                    mae=-0.05,
                )
            },
        ),
        _observation(
            "batch_c:candidate_c",
            [
                _decision(
                    "champion",
                    chosen_horizon=None,
                    final_percentile=0.99,
                )
            ],
            {
                "5": _label(
                    stock_return=0.50,
                    alpha=0.40,
                    mfe=0.60,
                    mae=-0.20,
                )
            },
        ),
        _observation(
            "batch_d:candidate_d",
            [
                _decision(
                    "champion",
                    chosen_horizon=5,
                    final_percentile=0.60,
                )
            ],
            {},
        ),
    ]
    _write_scorecard(runtime_dir, observations)

    result = analyze_forward_rank_thresholds(
        runtime_dir=runtime_dir,
        thresholds=(0.50, 0.80, 0.95),
    )

    assert result["interpretation"] == "diagnostic_only_not_portfolio_simulation"
    assert result["evidence_ready"] is True
    assert result["strategy_count"] == 1
    strategy = result["strategies"][0]
    assert strategy["decision_count"] == 4
    assert strategy["economically_eligible_decision_count"] == 3
    assert strategy["no_eligible_horizon_decision_count"] == 1
    assert strategy["matured_economically_eligible_decision_count"] == 2

    row_50 = strategy["thresholds"][0]
    assert row_50["rank_threshold"] == 0.50
    assert row_50["selected_decision_count"] == 3
    assert row_50["matured_selected_decision_count"] == 2
    assert row_50["pending_selected_decision_count"] == 1
    assert row_50["average_stock_return"] == pytest.approx(0.07)
    assert row_50["average_alpha"] == pytest.approx(0.035)
    assert row_50["positive_stock_return_rate"] == pytest.approx(1.0)
    assert row_50["positive_alpha_rate"] == pytest.approx(0.5)
    assert row_50["average_max_favorable_excursion"] == pytest.approx(0.12)
    assert row_50["average_max_adverse_excursion"] == pytest.approx(-0.045)

    row_80 = strategy["thresholds"][1]
    assert row_80["selected_decision_count"] == 1
    assert row_80["matured_selected_decision_count"] == 1
    assert row_80["average_stock_return"] == pytest.approx(0.10)
    assert row_80["average_alpha"] == pytest.approx(0.08)

    row_95 = strategy["thresholds"][2]
    assert row_95["selected_decision_count"] == 1
    assert row_95["matured_selected_decision_count"] == 1


def test_threshold_analysis_keeps_strategies_separate_and_reports_pending(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    observations = [
        _observation(
            "batch_a:candidate_a",
            [
                _decision("champion", chosen_horizon=5, final_percentile=0.70),
                _decision(
                    "shadow",
                    chosen_horizon=5,
                    final_percentile=1.0,
                    rank_threshold=0.90,
                    emitted=True,
                ),
            ],
            {},
        )
    ]
    _write_scorecard(runtime_dir, observations)

    result = analyze_forward_rank_thresholds(
        runtime_dir=runtime_dir,
        thresholds=(0.60, 0.95),
    )

    assert result["evidence_ready"] is False
    assert result["matured_threshold_point_count"] == 0
    assert [item["strategy_id"] for item in result["strategies"]] == [
        "champion",
        "shadow",
    ]
    champion = result["strategies"][0]
    assert champion["thresholds"][0]["selected_decision_count"] == 1
    assert champion["thresholds"][0]["pending_selected_decision_count"] == 1
    assert champion["thresholds"][1]["selected_decision_count"] == 0
    shadow = result["strategies"][1]
    assert shadow["observed_rank_thresholds"] == [0.90]
    assert shadow["current_emitted_decision_count"] == 1
    assert shadow["thresholds"][1]["selected_decision_count"] == 1


def test_threshold_analysis_rejects_invalid_thresholds(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_scorecard(runtime_dir, [])

    with pytest.raises(ValueError, match="between 0 and 1"):
        analyze_forward_rank_thresholds(
            runtime_dir=runtime_dir,
            thresholds=(1.1,),
        )
