from pathlib import Path

client_path = Path("src/stock_trading/contracts/client.py")
client = client_path.read_text(encoding="utf-8")
old = '''    def search_contract_transactions_page(\n        self,\n        award_search_id: str,\n        *,\n        modified_after: date,\n'''
new = '''    def search_contract_transactions_page(\n        self,\n        award_search_id: str,\n        *,\n        generated_award_id: str,\n        modified_after: date,\n'''
assert old in client
client = client.replace(old, new, 1)
old = '''        normalized = award_search_id.strip()\n        if not normalized:\n            raise ValueError("award_search_id must not be empty")\n        self._validate_search_window(modified_after, modified_before, page=page, limit=limit)\n        response = self._request_with_retry(\n'''
new = '''        normalized = award_search_id.strip()\n        normalized_generated = generated_award_id.strip()\n        if not normalized:\n            raise ValueError("award_search_id must not be empty")\n        if not normalized_generated:\n            raise ValueError("generated_award_id must not be empty")\n        self._validate_search_window(modified_after, modified_before, page=page, limit=limit)\n        response = self._request_with_retry(\n'''
assert old in client
client = client.replace(old, new, 1)
old = '''                    "award_type_codes": list(CONTRACT_TYPE_CODES),\n                    "award_ids": [normalized],\n                    "time_period": [\n'''
new = '''                    "award_type_codes": list(CONTRACT_TYPE_CODES),\n                    "award_ids": [normalized],\n                    "award_unique_id": normalized_generated,\n                    "time_period": [\n'''
assert old in client
client = client.replace(old, new, 1)
client_path.write_text(client, encoding="utf-8")

runner_path = Path("src/stock_trading/live/run_current_usaspending_shadow.py")
runner = runner_path.read_text(encoding="utf-8")
old = '''                tx_search_raw = usaspending_client.search_contract_transactions_page(\n                    award_search_id,\n                    modified_after=modified_after,\n'''
new = '''                tx_search_raw = usaspending_client.search_contract_transactions_page(\n                    award_search_id,\n                    generated_award_id=generated_award_id,\n                    modified_after=modified_after,\n'''
assert old in runner
runner = runner.replace(old, new, 1)
runner_path.write_text(runner, encoding="utf-8")

test_path = Path("tests/test_usaspending_shadow_forward.py")
test = test_path.read_text(encoding="utf-8")
old = '''    client.search_contract_transactions_page(\n        "W91TEST-26-C-0001",\n        modified_after=date(2026, 8, 22),\n'''
new = '''    client.search_contract_transactions_page(\n        "W91TEST-26-C-0001",\n        generated_award_id="CONT_AWD_TEST_9700_-NONE-_-NONE-",\n        modified_after=date(2026, 8, 22),\n'''
assert old in test
test = test.replace(old, new, 1)
old = '''    assert transactions["filters"]["award_ids"] == ["W91TEST-26-C-0001"]\n    assert "award_unique_id" not in transactions["filters"]\n    assert transactions["filters"]["time_period"][0]["date_type"] == "last_modified_date"\n'''
new = '''    assert transactions["filters"]["award_ids"] == ["W91TEST-26-C-0001"]\n    assert transactions["filters"]["award_unique_id"] == "CONT_AWD_TEST_9700_-NONE-_-NONE-"\n    assert transactions["filters"]["time_period"][0]["date_type"] == "last_modified_date"\n'''
assert old in test
test = test.replace(old, new, 1)
test_path.write_text(test, encoding="utf-8")
