from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from stock_trading.core import (
    RawRecord,
    SemanticAnnotation,
    SemanticDirection,
    Source,
    content_sha256,
)
from stock_trading.live.analyze_forward_thresholds import analyze_forward_rank_thresholds
from stock_trading.live.decision_diagnostics import (
    CandidateDecisionDiagnostic,
    FileStrategyDecisionDiagnosticStore,
    HorizonDecisionDiagnostic,
    StrategyDecisionDiagnostics,
)
from stock_trading.live.run_current_lda_shadow import (
    FileLdaShadowIntake,
    LdaShadowIntakeState,
    LdaShadowPending,
    poll_current_lda_shadow,
    select_lda_shadow_batch,
)
from stock_trading.storage import DuckDbEventStore


UTC = timezone.utc


class _FakeLdaClient:
    def __init__(self, pages: list[dict]) -> None:
        self.pages = pages
        self.calls = 0

    def fetch_filings_page(self, **kwargs) -> RawRecord:
        index = min(self.calls, len(self.pages) - 1)
        payload = self.pages[index]
        self.calls += 1
        content = json.dumps(payload)
        return RawRecord(
            source=Source.LDA,
            source_record_id=f"lda-page-{self.calls}",
            fetched_at=datetime(2026, 8, 20, 2, 0, tzinfo=UTC),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, source_text: str, *, context: str = "") -> SemanticAnnotation:
        self.calls += 1
        assert "Specific lobbying issues" in source_text
        assert context == "US federal lobbying disclosure"
        return SemanticAnnotation(
            topics=("GOVERNMENT.PROCUREMENT",),
            direction=SemanticDirection.NEUTRAL,
            novelty=0.4,
            importance=0.6,
            company_relevance=0.9,
            policy_relevance=0.8,
            confidence=0.95,
            model="test-qwen",
            extractor_version="semantic-v1",
            schema_version="semantic-v1",
        )


def _filing(
    filing_uuid: str,
    posted_at: str,
    *,
    client_id: int,
    client_name: str,
) -> dict:
    return {
        "filing_uuid": filing_uuid,
        "dt_posted": posted_at,
        "filing_year": 2026,
        "filing_period": "second_quarter",
        "client": {"id": client_id, "name": client_name},
        "registrant": {"name": "TEST REGISTRANT"},
        "income": "50000",
        "lobbying_activities": [
            {
                "general_issue_code": "CPT",
                "description": "Federal procurement policy discussion",
                "government_entities": [{"name": "Department of Commerce"}],
            }
        ],
    }


def _seed_modeled_identity(data_root: Path, experiment_dir: Path) -> None:
    (data_root / "manifests").mkdir(parents=True)
    (data_root / "manifests" / "sec_companies.jsonl").write_text(
        json.dumps(
            {
                "company_id": "cmp_microsoft",
                "issuer_names": ["MICROSOFT CORPORATION"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "training_rows.jsonl").write_text(
        json.dumps({"company_id": "cmp_microsoft"}) + "\n",
        encoding="utf-8",
    )


def test_lda_shadow_poll_is_isolated_semantic_and_idempotent(tmp_path) -> None:
    pytest.importorskip("duckdb")
    data_root = tmp_path / "data"
    runtime_dir = data_root / "runtime"
    experiment_dir = data_root / "experiments" / "model"
    _seed_modeled_identity(data_root, experiment_dir)

    authoritative = DuckDbEventStore(data_root / "normalized" / "events.duckdb")
    assert authoritative.count() == 0
    page = {
        "results": [
            _filing(
                "filing-mapped",
                "2026-08-19T18:00:00+00:00",
                client_id=101,
                client_name="Microsoft Corporation",
            ),
            _filing(
                "filing-unmapped",
                "2026-08-19T19:00:00+00:00",
                client_id=202,
                client_name="Unknown Private Client",
            ),
        ],
        "next": None,
    }
    client = _FakeLdaClient([page])
    extractor = _FakeExtractor()

    first = poll_current_lda_shadow(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        lda_client=client,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 20, 2, 0, tzinfo=UTC),
    )

    assert first.new_filings_seen == 2
    assert first.mapped_event_count == 1
    assert first.semantic_enriched_event_count == 1
    assert first.unmapped_filing_count == 1
    assert first.pending_events_added == 1
    assert first.pending_event_count == 1
    assert extractor.calls == 1
    assert authoritative.count() == 0

    shadow = DuckDbEventStore(runtime_dir / "lda_shadow" / "events.duckdb")
    stored = shadow.all_events()
    assert len(stored) == 1
    assert stored[0].company_id == "cmp_microsoft"
    assert stored[0].semantic is not None
    assert stored[0].semantic.model == "test-qwen"

    second = poll_current_lda_shadow(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        lda_client=client,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 20, 2, 5, tzinfo=UTC),
    )
    assert second.new_filings_seen == 0
    assert second.pending_events_added == 0
    assert second.pending_event_count == 1
    assert extractor.calls == 1
    assert authoritative.count() == 0


def test_lda_shadow_truncated_pagination_does_not_advance_cursor(tmp_path) -> None:
    pytest.importorskip("duckdb")
    data_root = tmp_path / "data"
    runtime_dir = data_root / "runtime"
    experiment_dir = data_root / "experiments" / "model"
    _seed_modeled_identity(data_root, experiment_dir)
    page = {
        "results": [
            _filing(
                "filing-one",
                "2026-08-19T18:00:00+00:00",
                client_id=101,
                client_name="Microsoft Corporation",
            )
        ],
        "next": "https://lda.gov/api/v1/filings/?page=2",
    }

    with pytest.raises(RuntimeError, match="max_pages reached"):
        poll_current_lda_shadow(
            data_root=data_root,
            experiment_dir=experiment_dir,
            runtime_dir=runtime_dir,
            lda_client=_FakeLdaClient([page]),  # type: ignore[arg-type]
            extractor=_FakeExtractor(),  # type: ignore[arg-type]
            as_of=datetime(2026, 8, 20, 2, 0, tzinfo=UTC),
            max_pages=1,
        )

    state = FileLdaShadowIntake(runtime_dir / "lda_shadow" / "intake.json").load()
    assert state.cursor is None
    assert state.pending == ()
    assert DuckDbEventStore(runtime_dir / "lda_shadow" / "events.duckdb").count() == 0


@dataclass
class _FakeResolver:
    def cycle_execution_date(self, as_of: datetime) -> date:
        return date(2026, 8, 20)

    def execution_date(self, public_time: datetime) -> date:
        if public_time.date() <= date(2026, 8, 18):
            return date(2026, 8, 19)
        if public_time.date() == date(2026, 8, 19) and public_time.hour >= 20:
            return date(2026, 8, 21)
        return date(2026, 8, 20)


def test_lda_shadow_selection_keeps_current_and_separates_stale_future(tmp_path) -> None:
    intake = FileLdaShadowIntake(tmp_path / "intake.json")
    intake._save(  # noqa: SLF001 - state-machine boundary test
        LdaShadowIntakeState(
            pending=(
                LdaShadowPending("evt-stale", "cmp-a", datetime(2026, 8, 18, 12, tzinfo=UTC)),
                LdaShadowPending("evt-current", "cmp-a", datetime(2026, 8, 19, 12, tzinfo=UTC)),
                LdaShadowPending("evt-future", "cmp-a", datetime(2026, 8, 19, 21, tzinfo=UTC)),
            )
        )
    )

    selection = select_lda_shadow_batch(
        intake,
        resolver=_FakeResolver(),  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 20, 2, tzinfo=UTC),
    )
    assert selection.selected_event_ids == ("evt-current",)
    assert selection.stale_event_ids == ("evt-stale",)
    assert selection.future_event_ids == ("evt-future",)
    assert intake.dispose_stale(selection.stale_event_ids) == 1
    state = intake.load()
    assert {item.event_id for item in state.pending} == {"evt-current", "evt-future"}
    assert state.stale_event_ids == ("evt-stale",)


def _strategy_diagnostic(candidate_id: str) -> StrategyDecisionDiagnostics:
    horizon = HorizonDecisionDiagnostic(
        horizon_sessions=5,
        expected_return=0.02,
        expected_alpha=0.01,
        expected_downside=0.03,
        probability_positive=0.6,
        raw_profit_score=0.1,
        profit_percentile=0.7,
        alpha_percentile=0.7,
        combined_signal=0.7,
        eligible=True,
        eligibility_reasons=(),
        required_feature_count=10,
        missing_feature_count=0,
        missing_feature_names=(),
    )
    decision = CandidateDecisionDiagnostic(
        candidate_id=candidate_id,
        company_id="cmp-a",
        security_id="sec-a",
        execution_date=date(2026, 8, 20),
        chosen_horizon=5,
        final_percentile=0.7,
        rank_threshold=0.95,
        emitted=False,
        rejection_reason="below_final_rank_threshold",
        horizons=(horizon,),
    )
    return StrategyDecisionDiagnostics(
        strategy_id="champion",
        candidate_count=1,
        emitted_opportunity_count=0,
        decisions=(decision,),
    )


def _compact_decision() -> dict:
    return {
        "strategy_id": "champion",
        "emitted": False,
        "rejection_reason": "below_final_rank_threshold",
        "chosen_horizon": 5,
        "final_percentile": 0.7,
        "rank_threshold": 0.95,
        "horizons": [],
    }


def test_threshold_analysis_can_filter_lda_from_legacy_sec_evidence(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    diagnostics_root = runtime_dir / "decision_diagnostics"
    store = FileStrategyDecisionDiagnosticStore(diagnostics_root)
    store.write(
        batch_id="batch_sec",
        as_of=datetime(2026, 8, 19, 18, tzinfo=UTC),
        target_execution_date=date(2026, 8, 20),
        diagnostics=(_strategy_diagnostic("candidate-sec"),),
    )
    store.write(
        batch_id="batch_lda",
        as_of=datetime(2026, 8, 19, 19, tzinfo=UTC),
        target_execution_date=date(2026, 8, 20),
        diagnostics=(_strategy_diagnostic("candidate-lda"),),
        evidence_source="lda_shadow",
    )
    observations = []
    for batch_id, candidate_id, alpha in (
        ("batch_sec", "candidate-sec", 0.01),
        ("batch_lda", "candidate-lda", 0.03),
    ):
        observations.append(
            {
                "observation_id": f"{batch_id}:{candidate_id}",
                "batch_id": batch_id,
                "candidate_id": candidate_id,
                "company_id": "cmp-a",
                "security_id": "sec-a",
                "execution_date": "2026-08-20",
                "strategy_decisions": [_compact_decision()],
                "realized_labels": {
                    "5": {
                        "stock_return": 0.04,
                        "alpha": alpha,
                        "max_favorable_excursion": 0.06,
                        "max_adverse_excursion": -0.02,
                    }
                },
                "matured_horizon_count": 1,
                "fully_matured": False,
            }
        )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "forward_scorecard.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-08-27T21:00:00+00:00",
                "last_completed_xnys_session": "2026-08-27",
                "observations": observations,
            }
        ),
        encoding="utf-8",
    )

    all_sources = analyze_forward_rank_thresholds(
        runtime_dir=runtime_dir,
        thresholds=(0.5,),
    )
    assert all_sources["evidence_source_counts"] == {"lda_shadow": 1, "sec_form4": 1}
    assert all_sources["included_observation_count"] == 2

    lda_only = analyze_forward_rank_thresholds(
        runtime_dir=runtime_dir,
        thresholds=(0.5,),
        evidence_source="lda_shadow",
    )
    assert lda_only["evidence_source_counts"] == {"lda_shadow": 1}
    assert lda_only["included_observation_count"] == 1
    assert lda_only["strategies"][0]["thresholds"][0]["average_alpha"] == pytest.approx(0.03)

    sec_only = analyze_forward_rank_thresholds(
        runtime_dir=runtime_dir,
        thresholds=(0.5,),
        evidence_source="sec_form4",
    )
    assert sec_only["included_observation_count"] == 1
    assert sec_only["strategies"][0]["thresholds"][0]["average_alpha"] == pytest.approx(0.01)


def test_diagnostic_evidence_source_rejects_empty_tag(tmp_path) -> None:
    with pytest.raises(ValueError, match="evidence_source"):
        FileStrategyDecisionDiagnosticStore(tmp_path).write(
            batch_id="batch_empty_source",
            as_of=datetime(2026, 8, 19, 18, tzinfo=UTC),
            target_execution_date=date(2026, 8, 20),
            diagnostics=(_strategy_diagnostic("candidate"),),
            evidence_source="   ",
        )
