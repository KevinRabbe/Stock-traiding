from pathlib import Path

runner_path = Path('src/stock_trading/live/run_current_usaspending_shadow.py')
test_path = Path('tests/test_usaspending_shadow_forward.py')
runner = runner_path.read_text(encoding='utf-8')
test = test_path.read_text(encoding='utf-8')

old = '''@dataclass(frozen=True, slots=True)\nclass UsaSpendingTransactionDiagnostic:\n    award_id: str\n    action_date: str\n    modification_number: str\n    transaction_amount: str\n    reason: str\n    match_count: int\n\n\nUSASPENDING_DIAGNOSTIC_LIMIT = 10\n'''
new = '''@dataclass(frozen=True, slots=True)\nclass UsaSpendingTransactionDiagnostic:\n    award_id: str\n    action_date: str\n    modification_number: str\n    transaction_amount: str\n    reason: str\n    match_count: int\n\n\n@dataclass(frozen=True, slots=True)\nclass UsaSpendingMappedEventDiagnostic:\n    event_id: str\n    company_id: str\n    modeled_company_name: str\n    recipient_name: str\n    award_id: str\n    transaction_id: str\n    action_date: str\n    public_time: datetime\n    modification_number: str\n    obligation_amount: str\n    total_obligation: str\n    potential_award_amount: str\n    agency: str\n    subagency: str\n    action_type: str\n    description: str\n    semantic_topics: tuple[str, ...]\n    semantic_direction: str\n    semantic_importance: float | None\n    semantic_company_relevance: float | None\n    semantic_confidence: float | None\n\n\nUSASPENDING_DIAGNOSTIC_LIMIT = 10\n'''
assert old in runner
runner = runner.replace(old, new, 1)

old = '''    matched_transaction_count: int\n    unmatched_transaction_count: int\n    transaction_diagnostic_sample: tuple[UsaSpendingTransactionDiagnostic, ...]\n    mapped_event_count: int\n    semantic_enriched_event_count: int\n'''
new = '''    matched_transaction_count: int\n    unmatched_transaction_count: int\n    transaction_diagnostic_sample: tuple[UsaSpendingTransactionDiagnostic, ...]\n    mapped_event_count: int\n    recent_mapped_event_sample: tuple[UsaSpendingMappedEventDiagnostic, ...]\n    semantic_enriched_event_count: int\n'''
assert old in runner
runner = runner.replace(old, new, 1)

old = '''    final_state = intake.load()\n    return UsaSpendingShadowPollResult(\n'''
new = '''    final_state = intake.load()\n    recent_mapped_event_sample = _recent_mapped_event_sample(\n        known_events.values(),\n        name_index=name_index,\n    )\n    return UsaSpendingShadowPollResult(\n'''
assert old in runner
runner = runner.replace(old, new, 1)

old = '''        unmatched_transaction_count=unmatched_transactions,\n        transaction_diagnostic_sample=tuple(transaction_sample),\n        mapped_event_count=len(enriched_events) + len(recovered_events),\n        semantic_enriched_event_count=len(enriched_events),\n'''
new = '''        unmatched_transaction_count=unmatched_transactions,\n        transaction_diagnostic_sample=tuple(transaction_sample),\n        mapped_event_count=len(enriched_events) + len(recovered_events),\n        recent_mapped_event_sample=recent_mapped_event_sample,\n        semantic_enriched_event_count=len(enriched_events),\n'''
assert old in runner
runner = runner.replace(old, new, 1)

marker = '''def _transaction_search_rows_for_award(\n    rows: list[dict],\n    generated_award_id: str,\n) -> tuple[list[dict], list[dict]]:\n'''
assert marker in runner
helper = '''def _recent_mapped_event_sample(\n    events: Iterable[Event],\n    *,\n    name_index: dict[str, tuple[str, ...]],\n    limit: int = USASPENDING_DIAGNOSTIC_LIMIT,\n) -> tuple[UsaSpendingMappedEventDiagnostic, ...]:\n    if limit <= 0:\n        return ()\n    eligible = [\n        event\n        for event in events\n        if event.source is Source.USASPENDING\n        and event.event_type is EventType.GOVERNMENT_CONTRACT\n        and event.company_id\n        and event.semantic is not None\n    ]\n    eligible.sort(\n        key=lambda event: (event.public_time, event.event_time, event.event_id),\n        reverse=True,\n    )\n    diagnostics: list[UsaSpendingMappedEventDiagnostic] = []\n    for event in eligible[:limit]:\n        payload = event.payload\n        semantic = event.semantic\n        assert semantic is not None\n        diagnostics.append(\n            UsaSpendingMappedEventDiagnostic(\n                event_id=event.event_id,\n                company_id=event.company_id or '',\n                modeled_company_name=_modeled_company_display_name(\n                    event.company_id or '',\n                    name_index,\n                ),\n                recipient_name=str(payload.recipient_name or ''),\n                award_id=str(payload.award_id or ''),\n                transaction_id=str(payload.transaction_id or ''),\n                action_date=event.event_time.date().isoformat(),\n                public_time=event.public_time,\n                modification_number=str(payload.modification_number or ''),\n                obligation_amount=_diagnostic_decimal(payload.obligation_amount),\n                total_obligation=_diagnostic_decimal(payload.total_obligation),\n                potential_award_amount=_diagnostic_decimal(payload.potential_award_amount),\n                agency=str(payload.agency or ''),\n                subagency=str(payload.subagency or ''),\n                action_type=str(payload.action_type or ''),\n                description=_diagnostic_description(payload.description),\n                semantic_topics=tuple(semantic.topics),\n                semantic_direction=semantic.direction.value,\n                semantic_importance=float(semantic.importance),\n                semantic_company_relevance=float(semantic.company_relevance),\n                semantic_confidence=float(semantic.confidence),\n            )\n        )\n    return tuple(diagnostics)\n\n\ndef _modeled_company_display_name(\n    company_id: str,\n    name_index: dict[str, tuple[str, ...]],\n) -> str:\n    names = [name for name, matches in name_index.items() if company_id in matches]\n    if not names:\n        return company_id\n    return min(names, key=lambda name: (len(name), name))\n\n\ndef _diagnostic_decimal(value) -> str:\n    return '' if value is None else str(value)\n\n\ndef _diagnostic_description(value, *, limit: int = 240) -> str:\n    text = str(value or '').strip()\n    if len(text) <= limit:\n        return text\n    return text[: max(0, limit - 3)].rstrip() + '...'\n\n\n'''
runner = runner.replace(marker, helper + marker, 1)

old = '''    assert result.mapped_event_count == 1\n    assert result.semantic_enriched_event_count == 1\n    assert result.pending_events_added == 1\n'''
new = '''    assert result.mapped_event_count == 1\n    assert len(result.recent_mapped_event_sample) == 1\n    mapped = result.recent_mapped_event_sample[0]\n    assert mapped.company_id == "cmp_microsoft"\n    assert mapped.modeled_company_name == "MICROSOFT CORPORATION"\n    assert mapped.recipient_name == "Microsoft Federal LLC"\n    assert mapped.award_id == "CONT_AWD_TEST_9700_-NONE-_-NONE-"\n    assert mapped.transaction_id == "CONT_TX_TEST_1"\n    assert mapped.action_date == "2026-08-22"\n    assert mapped.modification_number == "P00001"\n    assert mapped.obligation_amount == "250000"\n    assert mapped.agency == "Department of Defense"\n    assert mapped.semantic_topics == ("GOVERNMENT.PROCUREMENT",)\n    assert mapped.semantic_direction == "positive"\n    assert mapped.semantic_importance == 0.8\n    assert mapped.semantic_company_relevance == 0.95\n    assert mapped.semantic_confidence == 0.95\n    assert result.semantic_enriched_event_count == 1\n    assert result.pending_events_added == 1\n'''
assert old in test
test = test.replace(old, new, 1)

old = '''    assert second.pending_event_count == 1\n    assert second.semantic_enriched_event_count == 0\n    assert extractor.calls == 1\n'''
new = '''    assert second.pending_event_count == 1\n    assert second.semantic_enriched_event_count == 0\n    assert len(second.recent_mapped_event_sample) == 1\n    assert second.recent_mapped_event_sample[0].event_id == mapped.event_id\n    assert extractor.calls == 1\n'''
assert old in test
test = test.replace(old, new, 1)

runner_path.write_text(runner, encoding='utf-8')
test_path.write_text(test, encoding='utf-8')
