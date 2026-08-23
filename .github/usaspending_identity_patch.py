from pathlib import Path

runner_path = Path("src/stock_trading/live/run_current_usaspending_shadow.py")
runner = runner_path.read_text(encoding="utf-8")

old = '''    transaction_search_row_count: int\n    matched_transaction_count: int\n    unmatched_transaction_count: int\n'''
new = '''    transaction_search_row_count: int\n    transaction_identity_filtered_row_count: int\n    matched_transaction_count: int\n    unmatched_transaction_count: int\n'''
assert old in runner
runner = runner.replace(old, new, 1)

old = '''    transaction_search_rows = 0\n    matched_transactions = 0\n    unmatched_transactions = 0\n'''
new = '''    transaction_search_rows = 0\n    transaction_identity_filtered_rows = 0\n    matched_transactions = 0\n    unmatched_transactions = 0\n'''
assert old in runner
runner = runner.replace(old, new, 1)

old = '''                changed_rows.extend(item for item in tx_results if isinstance(item, dict))\n                transaction_search_rows += len(tx_results)\n                if not _has_next(tx_search_payload):\n'''
new = '''                typed_tx_results = [item for item in tx_results if isinstance(item, dict)]\n                accepted_rows, filtered_rows = _transaction_search_rows_for_award(\n                    typed_tx_results,\n                    generated_award_id,\n                )\n                changed_rows.extend(accepted_rows)\n                transaction_search_rows += len(typed_tx_results)\n                transaction_identity_filtered_rows += len(filtered_rows)\n                for filtered in filtered_rows:\n                    _append_transaction_diagnostic(\n                        transaction_sample,\n                        award_search_id,\n                        filtered,\n                        (\n                            "different_generated_award"\n                            if str(filtered.get("generated_internal_id") or "").strip()\n                            else "missing_generated_award_identity"\n                        ),\n                        0,\n                    )\n                if not _has_next(tx_search_payload):\n'''
assert old in runner
runner = runner.replace(old, new, 1)

old = '''        transaction_search_row_count=transaction_search_rows,\n        matched_transaction_count=matched_transactions,\n'''
new = '''        transaction_search_row_count=transaction_search_rows,\n        transaction_identity_filtered_row_count=transaction_identity_filtered_rows,\n        matched_transaction_count=matched_transactions,\n'''
assert old in runner
runner = runner.replace(old, new, 1)

anchor = '''def _matching_transaction_ids(search_row: dict, detail_rows: list[dict]) -> list[str]:\n'''
helper = '''def _transaction_search_rows_for_award(\n    rows: list[dict],\n    generated_award_id: str,\n) -> tuple[list[dict], list[dict]]:\n    """Partition transaction-search rows by exact USAspending award identity.\n\n    Display Award IDs/PIIDs can be reused beneath different parent vehicles.\n    The generated internal award ID is therefore the authority for deciding\n    whether a search row belongs to the award currently being normalized.\n    """\n\n    expected = generated_award_id.strip()\n    if not expected:\n        raise ValueError("generated_award_id must not be empty")\n    accepted: list[dict] = []\n    filtered: list[dict] = []\n    for row in rows:\n        observed = str(row.get("generated_internal_id") or "").strip()\n        if observed == expected:\n            accepted.append(row)\n        else:\n            filtered.append(row)\n    return accepted, filtered\n\n\n'''
assert anchor in runner
runner = runner.replace(anchor, helper + anchor, 1)
runner_path.write_text(runner, encoding="utf-8")

test_path = Path("tests/test_usaspending_shadow_forward.py")
test = test_path.read_text(encoding="utf-8")
old = '''    _matching_transaction_ids,\n    poll_current_usaspending_shadow,\n'''
new = '''    _matching_transaction_ids,\n    _transaction_search_rows_for_award,\n    poll_current_usaspending_shadow,\n'''
assert old in test
test = test.replace(old, new, 1)

anchor = '''def test_contract_search_client_uses_last_modified_and_documented_award_filter() -> None:\n'''
new_test = '''def test_transaction_search_rows_require_exact_generated_award_identity() -> None:\n    matching = {\n        "Award ID": "75F40125F19008",\n        "Mod": "P00002",\n        "generated_internal_id": "CONT_AWD_TARGET",\n    }\n    foreign = {\n        "Award ID": "75F40125F19008",\n        "Mod": "P00001",\n        "generated_internal_id": "CONT_AWD_OTHER_PARENT",\n    }\n    missing = {\n        "Award ID": "75F40125F19008",\n        "Mod": "P00003",\n    }\n\n    accepted, filtered = _transaction_search_rows_for_award(\n        [matching, foreign, missing],\n        "CONT_AWD_TARGET",\n    )\n\n    assert accepted == [matching]\n    assert filtered == [foreign, missing]\n\n\n'''
assert anchor in test
test = test.replace(anchor, new_test + anchor, 1)

old = '''    assert result.transaction_search_row_count == 1\n    assert result.matched_transaction_count == 1\n'''
new = '''    assert result.transaction_search_row_count == 1\n    assert result.transaction_identity_filtered_row_count == 0\n    assert result.matched_transaction_count == 1\n'''
assert old in test
test = test.replace(old, new, 1)
test_path.write_text(test, encoding="utf-8")
