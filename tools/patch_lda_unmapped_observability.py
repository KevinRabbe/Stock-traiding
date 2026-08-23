from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


live_path = Path("src/stock_trading/live/run_current_lda_shadow.py")
live = live_path.read_text(encoding="utf-8")

live = replace_once(
    live,
    '''@dataclass(frozen=True, slots=True)\nclass LdaShadowPollResult:\n    pages_fetched: int\n    filings_seen: int\n    new_filings_seen: int\n    mapped_event_count: int\n    semantic_enriched_event_count: int\n    unmapped_filing_count: int\n    nonmodeled_filing_count: int\n    pending_events_added: int\n    pending_event_count: int\n    cursor: LdaFilingCursor | None\n''',
    '''@dataclass(frozen=True, slots=True)\nclass LdaUnmappedFilingDiagnostic:\n    filing_uuid: str\n    posted_at: datetime\n    client_id: int | None\n    client_name: str\n    reason: str\n\n    def __post_init__(self) -> None:\n        object.__setattr__(self, "posted_at", as_utc(self.posted_at))\n        if not self.filing_uuid.strip():\n            raise ValueError("unmapped LDA filing diagnostic requires filing_uuid")\n        if self.reason not in {"invalid_client_object", "missing_client_identity", "unresolved_company"}:\n            raise ValueError(f"unsupported unmapped LDA filing reason: {self.reason}")\n\n\nLDA_UNMAPPED_DIAGNOSTIC_LIMIT = 10\n\n\n@dataclass(frozen=True, slots=True)\nclass LdaShadowPollResult:\n    pages_fetched: int\n    filings_seen: int\n    new_filings_seen: int\n    mapped_event_count: int\n    semantic_enriched_event_count: int\n    unmapped_filing_count: int\n    unmapped_filing_sample: tuple[LdaUnmappedFilingDiagnostic, ...]\n    nonmodeled_filing_count: int\n    pending_events_added: int\n    pending_event_count: int\n    cursor: LdaFilingCursor | None\n''',
    label="poll result dataclass",
)

live = replace_once(
    live,
    '''    unmapped = 0\n    nonmodeled = 0\n    enriched: list[Event] = []\n''',
    '''    unmapped = 0\n    unmapped_sample: list[LdaUnmappedFilingDiagnostic] = []\n    nonmodeled = 0\n    enriched: list[Event] = []\n\n    def record_unmapped(\n        cursor: LdaFilingCursor,\n        *,\n        client_id: int | None,\n        client_name: str,\n        reason: str,\n    ) -> None:\n        if len(unmapped_sample) >= LDA_UNMAPPED_DIAGNOSTIC_LIMIT:\n            return\n        unmapped_sample.append(\n            LdaUnmappedFilingDiagnostic(\n                filing_uuid=cursor.filing_uuid,\n                posted_at=cursor.posted_at,\n                client_id=client_id,\n                client_name=client_name,\n                reason=reason,\n            )\n        )\n''',
    label="poll counters",
)

live = replace_once(
    live,
    '''            client = filing.get("client") or {}\n            if not isinstance(client, dict):\n                unmapped += 1\n                continue\n''',
    '''            client = filing.get("client") or {}\n            if not isinstance(client, dict):\n                unmapped += 1\n                record_unmapped(\n                    cursor,\n                    client_id=None,\n                    client_name="",\n                    reason="invalid_client_object",\n                )\n                continue\n''',
    label="invalid client diagnostic",
)

live = replace_once(
    live,
    '''            client_id = _int_or_none(client.get("id"))\n            client_name = str(client.get("name") or "").strip()\n            if client_id is None or not client_name:\n                unmapped += 1\n                continue\n''',
    '''            client_id = _int_or_none(client.get("id"))\n            client_name = str(client.get("name") or "").strip()\n            if client_id is None or not client_name:\n                unmapped += 1\n                record_unmapped(\n                    cursor,\n                    client_id=client_id,\n                    client_name=client_name,\n                    reason="missing_client_identity",\n                )\n                continue\n''',
    label="missing client identity diagnostic",
)

live = replace_once(
    live,
    '''            if resolved is None:\n                unmapped += 1\n                continue\n''',
    '''            if resolved is None:\n                unmapped += 1\n                record_unmapped(\n                    cursor,\n                    client_id=client_id,\n                    client_name=client_name,\n                    reason="unresolved_company",\n                )\n                continue\n''',
    label="unresolved company diagnostic",
)

live = replace_once(
    live,
    '''        semantic_enriched_event_count=sum(item.semantic is not None for item in enriched),\n        unmapped_filing_count=unmapped,\n        nonmodeled_filing_count=nonmodeled,\n''',
    '''        semantic_enriched_event_count=sum(item.semantic is not None for item in enriched),\n        unmapped_filing_count=unmapped,\n        unmapped_filing_sample=tuple(unmapped_sample),\n        nonmodeled_filing_count=nonmodeled,\n''',
    label="poll result construction",
)

live_path.write_text(live, encoding="utf-8")


test_path = Path("tests/test_lda_shadow_forward.py")
tests = test_path.read_text(encoding="utf-8")

tests = replace_once(
    tests,
    '''    LdaShadowIntakeState,\n    LdaShadowPending,\n    poll_current_lda_shadow,\n''',
    '''    LdaShadowIntakeState,\n    LdaShadowPending,\n    LdaUnmappedFilingDiagnostic,\n    poll_current_lda_shadow,\n''',
    label="test import",
)

tests = replace_once(
    tests,
    '''    assert first.unmapped_filing_count == 1\n    assert first.pending_events_added == 1\n''',
    '''    assert first.unmapped_filing_count == 1\n    assert first.unmapped_filing_sample == (\n        LdaUnmappedFilingDiagnostic(\n            filing_uuid="filing-unmapped",\n            posted_at=datetime(2026, 8, 19, 19, 0, tzinfo=UTC),\n            client_id=202,\n            client_name="Unknown Private Client",\n            reason="unresolved_company",\n        ),\n    )\n    assert first.pending_events_added == 1\n''',
    label="existing unmapped assertion",
)

marker = '''\n\ndef test_lda_shadow_truncated_pagination_does_not_advance_cursor(tmp_path) -> None:\n'''
new_test = '''\n\ndef test_lda_shadow_unmapped_diagnostic_sample_is_bounded(tmp_path) -> None:\n    pytest.importorskip("duckdb")\n    data_root = tmp_path / "data"\n    runtime_dir = data_root / "runtime"\n    experiment_dir = data_root / "experiments" / "model"\n    _seed_modeled_identity(data_root, experiment_dir)\n    page = {\n        "results": [\n            _filing(\n                f"filing-unmapped-{index:02d}",\n                "2026-08-19T19:00:00+00:00",\n                client_id=1000 + index,\n                client_name=f"Unknown Client {index:02d}",\n            )\n            for index in range(12)\n        ],\n        "next": None,\n    }\n    extractor = _FakeExtractor()\n\n    result = poll_current_lda_shadow(\n        data_root=data_root,\n        experiment_dir=experiment_dir,\n        runtime_dir=runtime_dir,\n        lda_client=_FakeLdaClient([page]),  # type: ignore[arg-type]\n        extractor=extractor,  # type: ignore[arg-type]\n        as_of=datetime(2026, 8, 20, 2, 0, tzinfo=UTC),\n    )\n\n    assert result.unmapped_filing_count == 12\n    assert len(result.unmapped_filing_sample) == 10\n    assert result.unmapped_filing_sample[0].filing_uuid == "filing-unmapped-00"\n    assert result.unmapped_filing_sample[-1].filing_uuid == "filing-unmapped-09"\n    assert all(item.reason == "unresolved_company" for item in result.unmapped_filing_sample)\n    assert result.mapped_event_count == 0\n    assert extractor.calls == 0\n\n\ndef test_lda_shadow_truncated_pagination_does_not_advance_cursor(tmp_path) -> None:\n'''
tests = replace_once(tests, marker, new_test, label="bounded diagnostic test")

test_path.write_text(tests, encoding="utf-8")
