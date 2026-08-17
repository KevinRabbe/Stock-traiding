from __future__ import annotations

import pytest

from stock_trading.experiments.lightgbm_strategy_factory_executable_maturity import (
    SCHEMA_VERSION,
    _compact_console,
)
from stock_trading.experiments.lightgbm_strategy_qualify_executable_maturity import (
    _validate_report,
)


def _report() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": {
            "generation_id": "g002m",
            "completed_hypotheses": 48,
            "failed_hypotheses": 0,
        },
        "execution_realism": {
            "enabled": True,
            "full_fill_required": True,
            "return_cap_applied": False,
            "full_horizon_maturity_required": True,
            "maturity_fence": "latest_requested_horizon_exit_before_test_year",
        },
    }


def test_maturity_qualification_requires_explicit_maturity_fence() -> None:
    report = _report()
    _validate_report(report, "g002m")

    changed = {
        **report,
        "execution_realism": {
            **report["execution_realism"],
            "full_horizon_maturity_required": False,
        },
    }
    with pytest.raises(ValueError, match="full-horizon maturity"):
        _validate_report(changed, "g002m")


def test_maturity_compact_console_surfaces_fence() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation": {"generation_id": "g002m"},
        "execution_realism": {"full_horizon_maturity_required": True},
        "selection_policy": {"eligible_count": 3},
        "finalists": [],
        "failures": [],
    }

    compact = _compact_console(payload)

    assert compact["full_horizon_maturity_required"] is True
    assert compact["maturity_fence"] == "latest_requested_horizon_exit_before_test_year"
