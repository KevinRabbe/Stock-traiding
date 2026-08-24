from pathlib import Path

path = Path('.github/apply_usaspending_freshness.py')
text = path.read_text(encoding='utf-8')
old = '''replace_once(
    "tests/test_usaspending_shadow_forward.py",
    '\"Action Date\": \"2026-08-22\",\\n',
    '\"Action Date\": self.transaction_action_date,\\n',
)
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    '\"action_date\": \"2026-08-22\",\\n',
    '\"action_date\": self.transaction_action_date,\\n',
)
'''
new = '''replace_once(
    "tests/test_usaspending_shadow_forward.py",
    ''' + '"""' + '''                        \"Recipient UEI\": \"CHILDUEI1234\",\\n                        \"Action Date\": \"2026-08-22\",\\n                        \"Action Type\": \"FUNDING ONLY ACTION\",\\n''' + '"""' + ''',
    ''' + '"""' + '''                        \"Recipient UEI\": \"CHILDUEI1234\",\\n                        \"Action Date\": self.transaction_action_date,\\n                        \"Action Type\": \"FUNDING ONLY ACTION\",\\n''' + '"""' + ''',
)
replace_once(
    "tests/test_usaspending_shadow_forward.py",
    ''' + '"""' + '''                        \"type_description\": \"DEFINITIVE CONTRACT\",\\n                        \"action_date\": \"2026-08-22\",\\n                        \"action_type\": \"C\",\\n''' + '"""' + ''',
    ''' + '"""' + '''                        \"type_description\": \"DEFINITIVE CONTRACT\",\\n                        \"action_date\": self.transaction_action_date,\\n                        \"action_type\": \"C\",\\n''' + '"""' + ''',
)
'''
if old not in text:
    raise RuntimeError('ambiguous fixture patch block not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
