from __future__ import annotations

import pytest

from stock_trading.experiments.lightgbm_strategy_qualify_executable import (
    _assert_execution_diagnostics,
    _compact_console,
)


def test_execution_diagnostics_require_exact_replay_identity() -> None:
    expected = {
        "source_row_count": 12385,
        "quality_removed_row_count": 5,
        "pit_liquidity_removed_row_count": 380,
        "executable_row_count": 12000,
        "rejected_entry_liquidity": 14,
    }

    _assert_execution_diagnostics(expected, dict(expected))

    changed = dict(expected)
    changed["rejected_entry_liquidity"] = 13
    with pytest.raises(ValueError, match="execution diagnostic mismatch"):
        _assert_execution_diagnostics(expected, changed)


def test_execution_diagnostics_reject_missing_screening_field() -> None:
    expected = {
        "source_row_count": 12385,
        "quality_removed_row_count": 5,
        "pit_liquidity_removed_row_count": 380,
        "executable_row_count": 12000,
    }
    actual = {
        **expected,
        "rejected_entry_liquidity": 14,
    }

    with pytest.raises(ValueError, match="missing execution diagnostic"):
        _assert_execution_diagnostics(expected, actual)


def test_compact_console_surfaces_replay_and_concentration_flags() -> None:
    payload = {
        "schema_version": "lightgbm-strategy-finalist-qualification-executable-v1",
        "generation_id": "g002",
        "all_finalists_exactly_reproduced": True,
        "execution_realism": {
            "full_fill_required": True,
            "return_cap_applied": False,
        },
        "finalists": [
            {
                "variant_id": "factory-test",
                "scorecard": {"compounded_return": 0.10},
                "execution_diagnostics": {
                    "source_row_count": 12385,
                    "quality_removed_row_count": 5,
                    "pit_liquidity_removed_row_count": 380,
                    "executable_row_count": 12000,
                    "rejected_entry_liquidity": 14,
                },
                "qualification_flags": {
                    "best_year_dependency": False,
                    "top_three_year_dependency": False,
                },
                "diagnostics": {
                    "trade_count": 233,
                    "unique_company_count": 80,
                    "largest_positive_trade_pnl_fraction": 0.08,
                    "largest_positive_company_pnl_fraction": 0.12,
                    "best_three_years": [2016, 2019, 2020],
                    "compounded_return_excluding_best_three_years": 0.02,
                    "gross_return_distribution": {
                        "min": -0.4,
                        "median": 0.02,
                        "p95": 0.15,
                        "max": 0.8,
                    },
                },
            }
        ],
    }

    compact = _compact_console(payload)

    assert compact["all_finalists_exactly_reproduced"] is True
    assert compact["finalists"][0]["variant_id"] == "factory-test"
    assert compact["finalists"][0]["diagnostics"]["trade_count"] == 233
    assert compact["finalists"][0]["diagnostics"][
        "compounded_return_excluding_best_three_years"
    ] == pytest.approx(0.02)
