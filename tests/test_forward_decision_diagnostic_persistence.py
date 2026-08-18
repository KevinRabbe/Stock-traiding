from datetime import date, datetime, timezone

import pytest

from stock_trading.live.decision_diagnostics import (
    FileStrategyDecisionDiagnosticStore,
    StrategyDecisionDiagnostics,
    validate_diagnostic_counts,
)


UTC = timezone.utc


def _diagnostic(strategy_id: str, emitted: int) -> StrategyDecisionDiagnostics:
    return StrategyDecisionDiagnostics(
        strategy_id=strategy_id,
        candidate_count=1,
        emitted_opportunity_count=emitted,
        decisions=(),
    )


def test_diagnostic_counts_must_match_authoritative_strategy_results() -> None:
    diagnostics = (
        _diagnostic("champion", 1),
        _diagnostic("shadow-a", 0),
    )

    validate_diagnostic_counts(
        diagnostics,
        champion_strategy_id="champion",
        champion_opportunity_count=1,
        shadow_opportunity_counts={"shadow-a": 0},
    )

    with pytest.raises(RuntimeError, match="diagnostics disagree"):
        validate_diagnostic_counts(
            diagnostics,
            champion_strategy_id="champion",
            champion_opportunity_count=0,
            shadow_opportunity_counts={"shadow-a": 0},
        )


def test_decision_diagnostic_store_is_idempotent_and_immutable(tmp_path) -> None:
    store = FileStrategyDecisionDiagnosticStore(tmp_path / "decision_diagnostics")
    diagnostics = (_diagnostic("champion", 0),)
    kwargs = {
        "batch_id": "batch_123",
        "as_of": datetime(2026, 8, 18, 18, 0, tzinfo=UTC),
        "target_execution_date": date(2026, 8, 19),
        "diagnostics": diagnostics,
    }

    first = store.write(**kwargs)
    second = store.write(**kwargs)

    assert first == second
    assert first.is_file()

    with pytest.raises(ValueError, match="diagnostic changed"):
        store.write(
            batch_id="batch_123",
            as_of=datetime(2026, 8, 18, 18, 1, tzinfo=UTC),
            target_execution_date=date(2026, 8, 19),
            diagnostics=diagnostics,
        )
