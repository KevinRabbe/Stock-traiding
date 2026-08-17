from __future__ import annotations

import pytest

from stock_trading.engine import StrategyRecord, StrategyStage
from stock_trading.experiments.lightgbm_strategy_freeze_executable_maturity import (
    FACTORY_SCHEMA,
    MATURITY_FENCE,
    QUALIFICATION_SCHEMA,
    _preflight_registry,
    _select_qualified_variants,
    _validate_sources,
)


def _report() -> dict:
    realism = {
        "enabled": True,
        "full_fill_required": True,
        "full_horizon_maturity_required": True,
        "invalid_target_count": 5,
        "market_quality_manifest": "data/manifests/market_quality_verified.json",
        "maturity_fence": MATURITY_FENCE,
        "max_entry_day_participation_pct": 0.01,
        "max_trailing_adv_participation_pct": 0.01,
        "return_cap_applied": False,
        "verified_quality_exclusion_count": 1,
    }
    finalists = [
        {"variant_id": "a", "selection_score": 0.9},
        {"variant_id": "b", "selection_score": 0.8},
        {"variant_id": "c", "selection_score": 0.7},
        {"variant_id": "d", "selection_score": 0.6},
    ]
    results = [
        {
            "spec": {
                "variant_id": item["variant_id"],
                "feature_profile": "event_history",
                "training_window_years": 5,
                "tree_profile": "baseline",
                "horizons": [5, 20, 60],
                "alpha_rank_weight": 0.25,
                "seed": 42,
                "validation_top_fraction": 0.05,
                "calibration_window_days": 365,
                "max_expected_downside": 0.06,
            },
            "yearly_returns": {"2026": 0.01},
            "trade_candidate_ids": [],
            "execution_diagnostics": {"rejected_entry_liquidity": 0},
        }
        for item in finalists
    ]
    return {
        "schema_version": FACTORY_SCHEMA,
        "generation": {"generation_id": "g002m", "failed_hypotheses": 0},
        "execution_realism": realism,
        "portfolio_policy": {
            "starting_capital": 10_000.0,
            "allocation_pct": 0.02,
            "max_open_positions": 15,
            "round_trip_cost_bps": 20.0,
        },
        "finalists": finalists,
        "results": results,
    }


def _qualified(
    variant_id: str,
    *,
    flagged: bool = False,
    ex_best_three: float = 0.01,
) -> dict:
    return {
        "variant_id": variant_id,
        "exact_screening_identity_verified": True,
        "scorecard": {
            "compounded_return": 0.10,
            "profit_factor": 1.5,
            "worst_realized_drawdown": 0.02,
            "total_trades": 100,
            "profitable_year_rate": 0.7,
            "average_trade_alpha": 0.01,
        },
        "qualification_flags": {
            "best_year_dependency": False,
            "single_company_positive_pnl_concentration_ge_25pct": False,
            "single_trade_positive_pnl_concentration_ge_25pct": False,
            "top_three_year_dependency": flagged,
        },
        "diagnostics": {
            "compounded_return_excluding_best_three_years": ex_best_three,
        },
    }


def _qualification(report: dict) -> dict:
    realism = dict(report["execution_realism"])
    realism.pop("enabled")
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "generation_id": "g002m",
        "all_finalists_exactly_reproduced": True,
        "full_horizon_maturity_required": True,
        "execution_realism": realism,
        "finalists": [
            _qualified("a"),
            _qualified("b"),
            _qualified("c"),
            _qualified("d", flagged=True, ex_best_three=-0.01),
        ],
    }


def test_default_freeze_selection_keeps_first_three_clean_finalists() -> None:
    report = _report()
    qualification = _qualification(report)

    selected = _select_qualified_variants(
        report,
        qualification,
        variant_ids=(),
        finalist_count=3,
    )

    assert [item["variant_id"] for item in selected] == ["a", "b", "c"]


def test_explicit_freeze_selection_rejects_flagged_finalist() -> None:
    report = _report()
    qualification = _qualification(report)

    with pytest.raises(ValueError, match="did not pass clean qualification"):
        _select_qualified_variants(
            report,
            qualification,
            variant_ids=("d",),
            finalist_count=3,
        )


def test_source_validation_requires_full_horizon_maturity_identity() -> None:
    report = _report()
    qualification = _qualification(report)
    _validate_sources(report, qualification, "g002m")

    changed = dict(qualification)
    changed["execution_realism"] = {
        **qualification["execution_realism"],
        "maturity_fence": "20d_only",
    }
    with pytest.raises(ValueError, match="differs"):
        _validate_sources(report, changed, "g002m")


def test_source_validation_refuses_non_maturity_factory_report() -> None:
    report = _report()
    qualification = _qualification(report)
    changed = {
        **report,
        "execution_realism": {
            **report["execution_realism"],
            "full_horizon_maturity_required": False,
        },
    }

    with pytest.raises(ValueError, match="full-horizon"):
        _validate_sources(changed, qualification, "g002m")


def test_registry_preflight_refuses_existing_selected_strategy() -> None:
    report = _report()
    qualification = _qualification(report)
    selected = _select_qualified_variants(
        report,
        qualification,
        variant_ids=("a",),
        finalist_count=3,
    )
    existing = {
        "a": StrategyRecord(strategy_id="a", stage=StrategyStage.SHADOW),
    }

    with pytest.raises(RuntimeError, match="refuses overwrite"):
        _preflight_registry(existing, selected)
