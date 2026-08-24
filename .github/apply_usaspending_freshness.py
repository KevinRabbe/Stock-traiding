from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match in {path}, found {count}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Make invalidation writes crash-idempotent: timestamp differences do not change meaning.
replace_once(
    "src/stock_trading/live/forward_evidence_invalidations.py",
    """            if previous is not None:\n                if previous != item:\n                    raise ValueError(\n                        f\"forward evidence invalidation changed for {item.batch_id}\"\n                    )\n                continue\n""",
    """            if previous is not None:\n                if (\n                    previous.evidence_source != item.evidence_source\n                    or previous.reason != item.reason\n                ):\n                    raise ValueError(\n                        f\"forward evidence invalidation changed for {item.batch_id}\"\n                    )\n                continue\n""",
)

# Forward scorecards must ignore invalidated immutable diagnostics while retaining audit files.
replace_once(
    "src/stock_trading/live/forward_outcomes.py",
    "from .run_current_paper_shadow import _load_runtime_config\n",
    "from .forward_evidence_invalidations import FileForwardEvidenceInvalidationStore\nfrom .run_current_paper_shadow import _load_runtime_config\n",
)
replace_once(
    "src/stock_trading/live/forward_outcomes.py",
    """    instances = _load_forward_decision_instances(runtime_dir / \"decision_diagnostics\")\n""",
    """    diagnostic_root = runtime_dir / \"decision_diagnostics\"\n    invalidated_batch_ids = FileForwardEvidenceInvalidationStore(\n        runtime_dir / \"forward_evidence_invalidations.json\"\n    ).invalidated_batch_ids()\n    invalidated_diagnostic_batch_count = sum(\n        1\n        for path in diagnostic_root.glob(\"batch_*.json\")\n        if path.stem in invalidated_batch_ids\n    )\n    instances = _load_forward_decision_instances(\n        diagnostic_root,\n        invalidated_batch_ids=invalidated_batch_ids,\n    )\n""",
)
replace_once(
    "src/stock_trading/live/forward_outcomes.py",
    """            \"diagnostic_batch_count\": len({item.batch_id for item in instances}),\n""",
    """            \"diagnostic_batch_count\": len({item.batch_id for item in instances}),\n            \"invalidated_diagnostic_batch_count\": invalidated_diagnostic_batch_count,\n""",
)
replace_once(
    "src/stock_trading/live/forward_outcomes.py",
    """def _load_forward_decision_instances(root: Path) -> tuple[ForwardDecisionInstance, ...]:\n    if not root.exists():\n        return ()\n\n    result: list[ForwardDecisionInstance] = []\n    for path in sorted(root.glob(\"batch_*.json\")):\n""",
    """def _load_forward_decision_instances(\n    root: Path,\n    *,\n    invalidated_batch_ids: frozenset[str] = frozenset(),\n) -> tuple[ForwardDecisionInstance, ...]:\n    if not root.exists():\n        return ()\n\n    result: list[ForwardDecisionInstance] = []\n    for path in sorted(root.glob(\"batch_*.json\")):\n        if path.stem in invalidated_batch_ids:\n            continue\n""",
)

# USAspending source freshness: last_modified_date is discovery only; action_date is economic time.
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    "from .forward_outcomes import refresh_forward_outcome_scorecard\n",
    """from .forward_evidence_invalidations import (\n    FileForwardEvidenceInvalidationStore,\n    ForwardEvidenceInvalidation,\n)\nfrom .forward_outcomes import refresh_forward_outcome_scorecard\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    "USASPENDING_DIAGNOSTIC_LIMIT = 10\n",
    """USASPENDING_DIAGNOSTIC_LIMIT = 10\nUSASPENDING_MAX_ACTION_LAG_DAYS = 30\nUSASPENDING_FRESHNESS_MIGRATION_SCHEMA_VERSION = 1\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """    transaction_search_row_count: int\n    transaction_identity_filtered_row_count: int\n    matched_transaction_count: int\n""",
    """    transaction_search_row_count: int\n    transaction_identity_filtered_row_count: int\n    freshness_filtered_transaction_count: int\n    matched_transaction_count: int\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """    event_store = DuckDbEventStore(shadow_root / \"events.duckdb\")\n    known_events = {event.event_id: event for event in event_store.all_events()}\n""",
    """    event_store = DuckDbEventStore(shadow_root / \"events.duckdb\")\n    invalidated_event_ids = _load_usaspending_freshness_invalidated_event_ids(shadow_root)\n    known_events = {\n        event.event_id: event\n        for event in event_store.all_events()\n        if event.event_id not in invalidated_event_ids\n    }\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """    transaction_search_rows = 0\n    transaction_identity_filtered_rows = 0\n    matched_transactions = 0\n""",
    """    transaction_search_rows = 0\n    transaction_identity_filtered_rows = 0\n    freshness_filtered_transactions = 0\n    matched_transactions = 0\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """                changed_rows.extend(accepted_rows)\n                transaction_search_rows += len(typed_tx_results)\n                transaction_identity_filtered_rows += len(filtered_rows)\n                for filtered in filtered_rows:\n""",
    """                fresh_rows: list[dict] = []\n                for accepted in accepted_rows:\n                    freshness_reason = _transaction_action_freshness_reason(\n                        accepted,\n                        observed_on=cutoff.date(),\n                    )\n                    if freshness_reason is None:\n                        fresh_rows.append(accepted)\n                    else:\n                        freshness_filtered_transactions += 1\n                        _append_transaction_diagnostic(\n                            transaction_sample,\n                            award_search_id,\n                            accepted,\n                            freshness_reason,\n                            0,\n                        )\n                changed_rows.extend(fresh_rows)\n                transaction_search_rows += len(typed_tx_results)\n                transaction_identity_filtered_rows += len(filtered_rows)\n                for filtered in filtered_rows:\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """        transaction_search_row_count=transaction_search_rows,\n        transaction_identity_filtered_row_count=transaction_identity_filtered_rows,\n        matched_transaction_count=matched_transactions,\n""",
    """        transaction_search_row_count=transaction_search_rows,\n        transaction_identity_filtered_row_count=transaction_identity_filtered_rows,\n        freshness_filtered_transaction_count=freshness_filtered_transactions,\n        matched_transaction_count=matched_transactions,\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """    intake = FileUsaSpendingShadowIntake(shadow_root / \"intake.json\")\n\n    try:\n""",
    """    intake = FileUsaSpendingShadowIntake(shadow_root / \"intake.json\")\n    freshness_reconciliation = _reconcile_usaspending_action_freshness_v1(\n        runtime_root,\n        cutoff=cutoff,\n    )\n\n    try:\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """        \"qwen_model\": resolved_qwen_model,\n        \"poll\": _jsonable(asdict(poll)),\n""",
    """        \"qwen_model\": resolved_qwen_model,\n        \"freshness_policy\": {\n            \"max_action_lag_days\": USASPENDING_MAX_ACTION_LAG_DAYS,\n            \"discovery_date_type\": \"last_modified_date\",\n            \"economic_date_type\": \"action_date\",\n        },\n        \"freshness_reconciliation\": freshness_reconciliation,\n        \"poll\": _jsonable(asdict(poll)),\n""",
)
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """    histories: list[Iterable[Event]] = [authoritative_events]\n    for path in (\n        runtime_root / \"lda_shadow\" / \"events.duckdb\",\n        shadow_root / \"events.duckdb\",\n    ):\n        if path.exists():\n            histories.append(DuckDbEventStore(path).all_events(company_ids=list(ready_set)))\n""",
    """    histories: list[Iterable[Event]] = [authoritative_events]\n    invalidated_usaspending_event_ids = _load_usaspending_freshness_invalidated_event_ids(\n        shadow_root\n    )\n    for path in (\n        runtime_root / \"lda_shadow\" / \"events.duckdb\",\n        shadow_root / \"events.duckdb\",\n    ):\n        if path.exists():\n            history_events = DuckDbEventStore(path).all_events(company_ids=list(ready_set))\n            if path == shadow_root / \"events.duckdb\":\n                history_events = tuple(\n                    event\n                    for event in history_events\n                    if event.event_id not in invalidated_usaspending_event_ids\n                )\n            histories.append(history_events)\n""",
)

# Freshness and one-time reconciliation helpers.
replace_once(
    "src/stock_trading/live/run_current_usaspending_shadow.py",
    """def _transaction_search_rows_for_award(\n""",
    """def _transaction_action_freshness_reason(\n    row: dict,\n    *,\n    observed_on: date,\n    max_action_lag_days: int = USASPENDING_MAX_ACTION_LAG_DAYS,\n) -> str | None:\n    if max_action_lag_days < 0:\n        raise ValueError(\"max_action_lag_days must be >= 0\")\n    text = str(row.get(\"Action Date\") or \"\").strip()[:10]\n    try:\n        action_date = date.fromisoformat(text)\n    except ValueError:\n        return \"invalid_action_date\"\n    lag_days = (observed_on - action_date).days\n    if lag_days < 0:\n        return \"future_action_date\"\n    if lag_days > max_action_lag_days:\n        return \"stale_action_date\"\n    return None\n\n\ndef _load_usaspending_freshness_migration(shadow_root: Path) -> dict | None:\n    path = shadow_root / \"action_freshness_v1.json\"\n    if not path.exists():\n        return None\n    try:\n        payload = json.loads(path.read_text(encoding=\"utf-8\"))\n    except (OSError, json.JSONDecodeError) as exc:\n        raise ValueError(f\"invalid USAspending action freshness migration: {path}\") from exc\n    if (\n        not isinstance(payload, dict)\n        or payload.get(\"schema_version\") != USASPENDING_FRESHNESS_MIGRATION_SCHEMA_VERSION\n    ):\n        raise ValueError(\"unsupported USAspending action freshness migration schema\")\n    event_ids = payload.get(\"invalidated_event_ids\")\n    batch_ids = payload.get(\"invalidated_forward_batch_ids\")\n    if not isinstance(event_ids, list) or not isinstance(batch_ids, list):\n        raise ValueError(\"invalid USAspending action freshness migration payload\")\n    return payload\n\n\ndef _load_usaspending_freshness_invalidated_event_ids(shadow_root: Path) -> frozenset[str]:\n    payload = _load_usaspending_freshness_migration(shadow_root)\n    if payload is None:\n        return frozenset()\n    return frozenset(str(item) for item in payload[\"invalidated_event_ids\"])\n\n\ndef _reconcile_usaspending_action_freshness_v1(\n    runtime_root: Path,\n    *,\n    cutoff: datetime,\n) -> dict:\n    shadow_root = runtime_root / \"usaspending_shadow\"\n    existing = _load_usaspending_freshness_migration(shadow_root)\n    if existing is not None:\n        return {\n            \"applied_now\": False,\n            \"invalidated_stored_event_count\": len(existing[\"invalidated_event_ids\"]),\n            \"invalidated_forward_batch_count\": len(existing[\"invalidated_forward_batch_ids\"]),\n            \"invalidated_event_ids\": list(existing[\"invalidated_event_ids\"]),\n            \"invalidated_forward_batch_ids\": list(existing[\"invalidated_forward_batch_ids\"]),\n        }\n\n    event_path = shadow_root / \"events.duckdb\"\n    stale_events: list[Event] = []\n    if event_path.exists():\n        for event in DuckDbEventStore(event_path).all_events():\n            if event.source is not Source.USASPENDING:\n                continue\n            lag_days = (event.public_time.date() - event.event_time.date()).days\n            if lag_days > USASPENDING_MAX_ACTION_LAG_DAYS:\n                stale_events.append(event)\n    invalidated_event_ids = tuple(sorted(event.event_id for event in stale_events))\n\n    invalidated_forward_batch_ids: tuple[str, ...] = ()\n    if invalidated_event_ids:\n        diagnostic_root = runtime_root / \"decision_diagnostics\"\n        invalidations: list[ForwardEvidenceInvalidation] = []\n        for path in sorted(diagnostic_root.glob(\"batch_*.json\")):\n            try:\n                payload = json.loads(path.read_text(encoding=\"utf-8\"))\n            except (OSError, json.JSONDecodeError) as exc:\n                raise ValueError(f\"invalid forward decision diagnostic: {path}\") from exc\n            if not isinstance(payload, dict) or payload.get(\"schema_version\") != 1:\n                raise ValueError(f\"unsupported forward decision diagnostic: {path}\")\n            if str(payload.get(\"evidence_source\") or \"\").strip() != \"usaspending_shadow\":\n                continue\n            batch = str(payload.get(\"batch_id\") or \"\")\n            if batch != path.stem:\n                raise ValueError(f\"forward diagnostic batch identity mismatch: {path}\")\n            invalidations.append(\n                ForwardEvidenceInvalidation(\n                    batch_id=batch,\n                    evidence_source=\"usaspending_shadow\",\n                    reason=\"usaspending_pre_action_freshness_guard_v1\",\n                    invalidated_at=cutoff,\n                )\n            )\n        FileForwardEvidenceInvalidationStore(\n            runtime_root / \"forward_evidence_invalidations.json\"\n        ).add_many(tuple(invalidations))\n        invalidated_forward_batch_ids = tuple(sorted(item.batch_id for item in invalidations))\n        FileUsaSpendingShadowIntake(shadow_root / \"intake.json\").dispose_stale(\n            invalidated_event_ids\n        )\n\n    migration_payload = {\n        \"schema_version\": USASPENDING_FRESHNESS_MIGRATION_SCHEMA_VERSION,\n        \"migrated_at\": cutoff.isoformat(),\n        \"max_action_lag_days\": USASPENDING_MAX_ACTION_LAG_DAYS,\n        \"invalidated_event_ids\": list(invalidated_event_ids),\n        \"invalidated_forward_batch_ids\": list(invalidated_forward_batch_ids),\n    }\n    _atomic_json_write(shadow_root / \"action_freshness_v1.json\", migration_payload)\n    return {\n        \"applied_now\": True,\n        \"invalidated_stored_event_count\": len(invalidated_event_ids),\n        \"invalidated_forward_batch_count\": len(invalidated_forward_batch_ids),\n        \"invalidated_event_ids\": list(invalidated_event_ids),\n        \"invalidated_forward_batch_ids\": list(invalidated_forward_batch_ids),\n    }\n\n\ndef _transaction_search_rows_for_award(\n""",
)

# Tests: stale transaction discovery is filtered before detail fetch/Qwen.
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    """    _matching_transaction_ids,\n    _transaction_search_rows_for_award,\n    poll_current_usaspending_shadow,\n""",
    """    _matching_transaction_ids,\n    _transaction_action_freshness_reason,\n    _transaction_search_rows_for_award,\n    poll_current_usaspending_shadow,\n""",
)
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    """    def __init__(self, *, award_has_next: bool = False) -> None:\n        self.award_has_next = award_has_next\n""",
    """    def __init__(\n        self,\n        *,\n        award_has_next: bool = False,\n        transaction_action_date: str = \"2026-08-22\",\n    ) -> None:\n        self.award_has_next = award_has_next\n        self.transaction_action_date = transaction_action_date\n""",
)
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    '"Action Date": "2026-08-22",\n',
    '"Action Date": self.transaction_action_date,\n',
)
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    '"action_date": "2026-08-22",\n',
    '"action_date": self.transaction_action_date,\n',
)
insert_after = """def test_transaction_search_rows_require_exact_generated_award_identity() -> None:\n"""
new_test = """def test_transaction_action_freshness_rejects_old_database_updates() -> None:\n    row = {\"Action Date\": \"2025-03-17\"}\n    assert (\n        _transaction_action_freshness_reason(\n            row,\n            observed_on=date(2026, 8, 23),\n        )\n        == \"stale_action_date\"\n    )\n    assert (\n        _transaction_action_freshness_reason(\n            {\"Action Date\": \"2026-08-22\"},\n            observed_on=date(2026, 8, 23),\n        )\n        is None\n    )\n\n\n"""
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    insert_after,
    new_test + insert_after,
)
# Add a full poll regression before the primary mapping test.
marker = """def test_usaspending_shadow_maps_explicit_parent_and_isolated_qwen(tmp_path) -> None:\n"""
stale_poll_test = """def test_usaspending_shadow_filters_stale_last_modified_transaction_before_qwen(tmp_path) -> None:\n    pytest.importorskip(\"duckdb\")\n    data_root = tmp_path / \"data\"\n    runtime_dir = data_root / \"runtime\"\n    experiment_dir = data_root / \"experiments\" / \"model\"\n    _seed_modeled_identity(data_root, experiment_dir)\n    extractor = _FakeExtractor()\n    client = _FakeUsaSpendingClient(transaction_action_date=\"2025-03-17\")\n\n    result = poll_current_usaspending_shadow(\n        data_root=data_root,\n        experiment_dir=experiment_dir,\n        runtime_dir=runtime_dir,\n        usaspending_client=client,  # type: ignore[arg-type]\n        extractor=extractor,  # type: ignore[arg-type]\n        as_of=datetime(2026, 8, 23, 19, 0, tzinfo=UTC),\n    )\n\n    assert result.mapped_award_count == 1\n    assert result.transaction_search_row_count == 1\n    assert result.freshness_filtered_transaction_count == 1\n    assert result.matched_transaction_count == 0\n    assert result.mapped_event_count == 0\n    assert result.pending_events_added == 0\n    assert client.transaction_detail_calls == 0\n    assert extractor.calls == 0\n    assert result.transaction_diagnostic_sample[0].reason == \"stale_action_date\"\n\n\n"""
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    marker,
    stale_poll_test + marker,
)
# Existing happy path should explicitly prove nothing was freshness-filtered.
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    """    assert result.transaction_identity_filtered_row_count == 0\n    assert result.matched_transaction_count == 1\n""",
    """    assert result.transaction_identity_filtered_row_count == 0\n    assert result.freshness_filtered_transaction_count == 0\n    assert result.matched_transaction_count == 1\n""",
)

# Forward scorecard invalidation regression.
replace_once(
    "tests/test_forward_outcomes.py",
    """from stock_trading.live.forward_outcomes import refresh_forward_outcome_scorecard\n""",
    """from stock_trading.live.forward_evidence_invalidations import (\n    FileForwardEvidenceInvalidationStore,\n    ForwardEvidenceInvalidation,\n)\nfrom stock_trading.live.forward_outcomes import refresh_forward_outcome_scorecard\n""",
)
append_test = r'''


def test_forward_scorecard_excludes_invalidated_diagnostic_batch(tmp_path) -> None:
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
    FileForwardEvidenceInvalidationStore(
        runtime_dir / "forward_evidence_invalidations.json"
    ).add_many(
        (
            ForwardEvidenceInvalidation(
                batch_id="batch_one",
                evidence_source="usaspending_shadow",
                reason="source_data_correction",
                invalidated_at=datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
            ),
        )
    )

    result = refresh_forward_outcome_scorecard(
        data_root=tmp_path / "data",
        runtime_dir=runtime_dir,
        as_of=datetime(2026, 8, 25, 21, 0, tzinfo=UTC),
    )

    assert result["diagnostic_batch_count"] == 1
    assert result["invalidated_diagnostic_batch_count"] == 1
    assert result["candidate_decision_instance_count"] == 1
    scorecard = json.loads((runtime_dir / "forward_scorecard.json").read_text(encoding="utf-8"))
    assert [item["batch_id"] for item in scorecard["observations"]] == ["batch_two"]
'''
p = Path("tests/test_forward_outcomes.py")
p.write_text(p.read_text(encoding="utf-8") + append_test, encoding="utf-8")
