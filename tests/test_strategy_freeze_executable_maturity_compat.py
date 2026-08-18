from __future__ import annotations

import pytest

from stock_trading.experiments.lightgbm_strategy_freeze_executable_maturity import (
    FACTORY_SCHEMA,
    MATURITY_FENCE,
    QUALIFICATION_SCHEMA,
)
from stock_trading.experiments.lightgbm_strategy_freeze_executable_maturity_compat import (
    _validate_sources_compatible,
)


def _report() -> dict:
    return {
        "schema_version": FACTORY_SCHEMA,
        "generation": {"generation_id": "g002m", "failed_hypotheses": 0},
        "execution_realism": {
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
        },
    }


def _persisted_qualification(report: dict) -> dict:
    realism = dict(report["execution_realism"])
    realism.pop("enabled")
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "generation_id": "g002m",
        "all_finalists_exactly_reproduced": True,
        # Deliberately no redundant top-level full_horizon_maturity_required:
        # this matches the qualifier's actual persisted qualification.json.
        "execution_realism": realism,
        "finalists": [],
    }


def test_accepts_actual_persisted_g002m_qualification_shape() -> None:
    report = _report()
    qualification = _persisted_qualification(report)

    _validate_sources_compatible(report, qualification, "g002m")


def test_still_rejects_maturity_fence_mismatch() -> None:
    report = _report()
    qualification = _persisted_qualification(report)
    qualification["execution_realism"] = {
        **qualification["execution_realism"],
        "maturity_fence": "20d_only",
    }

    with pytest.raises(ValueError, match="differs"):
        _validate_sources_compatible(report, qualification, "g002m")


def test_still_rejects_missing_authoritative_maturity_marker() -> None:
    report = _report()
    qualification = _persisted_qualification(report)
    qualification["execution_realism"] = {
        **qualification["execution_realism"],
        "full_horizon_maturity_required": False,
    }

    with pytest.raises(ValueError, match="differs"):
        _validate_sources_compatible(report, qualification, "g002m")
