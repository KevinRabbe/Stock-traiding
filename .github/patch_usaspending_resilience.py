from pathlib import Path

path = Path("src/stock_trading/live/run_current_usaspending_shadow.py")
text = path.read_text(encoding="utf-8")
old = '''def _possibly_modeled_recipient(
    recipient_name: str,
    name_index: dict[str, tuple[str, ...]],
) -> bool:
    """Broad acquisition prefilter only; never grants mapping authority."""

    candidate = normalize_company_name(recipient_name)
    if not candidate:
        return False
    candidate_tokens = {token for token in candidate.split() if len(token) >= 4}
    if not candidate_tokens:
        return False
    for modeled_name in name_index:
        if candidate == modeled_name:
            return True
        modeled_tokens = {token for token in modeled_name.split() if len(token) >= 4}
        shared = candidate_tokens & modeled_tokens
        if shared and (candidate.startswith(modeled_name) or modeled_name.startswith(candidate)):
            return True
        if any(len(token) >= 6 for token in shared):
            return True
    return False
'''
new = '''GENERIC_RECIPIENT_TOKENS = frozenset(
    {
        "AMERICAN",
        "COMPANY",
        "CORPORATION",
        "FEDERAL",
        "GENERAL",
        "GLOBAL",
        "GROUP",
        "HOLDING",
        "HOLDINGS",
        "INDUSTRIES",
        "INDUSTRY",
        "INTERNATIONAL",
        "LIMITED",
        "NATIONAL",
        "SERVICES",
        "SERVICE",
        "SOLUTIONS",
        "SYSTEMS",
        "TECHNOLOGIES",
        "TECHNOLOGY",
        "UNITED",
    }
)


def _possibly_modeled_recipient(
    recipient_name: str,
    name_index: dict[str, tuple[str, ...]],
) -> bool:
    """Conservative detail-fetch prefilter; never grants company identity."""

    candidate = normalize_company_name(recipient_name)
    if not candidate:
        return False
    candidate_tokens = _distinctive_recipient_tokens(candidate)
    if not candidate_tokens:
        return False
    for modeled_name in name_index:
        if candidate == modeled_name:
            return True
        if candidate.startswith(modeled_name) or modeled_name.startswith(candidate):
            return True
        modeled_tokens = _distinctive_recipient_tokens(modeled_name)
        if not modeled_tokens:
            continue
        # A subsidiary such as "Microsoft Federal" can justify a detail fetch
        # for modeled "Microsoft", but generic roots such as "National" cannot.
        if candidate_tokens[0] == modeled_tokens[0] and len(candidate_tokens[0]) >= 6:
            return True
    return False


def _distinctive_recipient_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in value.split()
        if len(token) >= 4 and token not in GENERIC_RECIPIENT_TOKENS
    )
'''
if text.count(old) != 1:
    raise SystemExit("expected USAspending prefilter block exactly once")
path.write_text(text.replace(old, new), encoding="utf-8")
