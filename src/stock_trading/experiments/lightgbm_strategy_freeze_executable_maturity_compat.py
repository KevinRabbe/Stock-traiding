from __future__ import annotations

from typing import Any, Mapping

from . import lightgbm_strategy_freeze_executable_maturity as freeze


def _validate_sources_compatible(
    report: Mapping[str, Any],
    qualification: Mapping[str, Any],
    generation_id: str,
) -> None:
    """Validate the actual persisted G002m qualification schema.

    The maturity qualifier persists the authoritative maturity marker inside
    ``execution_realism``. Its compact console output also displays a redundant
    top-level ``full_horizon_maturity_required`` field, but that field is not
    written to ``qualification.json``. The original freeze gate incorrectly
    required both representations and therefore rejected valid qualifications.
    """

    if report.get("schema_version") != freeze.FACTORY_SCHEMA:
        raise ValueError("freeze requires a G002m maturity-safe factory report")
    generation = report.get("generation") or {}
    if generation.get("generation_id") != generation_id:
        raise ValueError("factory report generation_id mismatch")
    if int(generation.get("failed_hypotheses", -1)) != 0:
        raise ValueError("freeze requires a generation with zero failed hypotheses")

    realism = report.get("execution_realism") or {}
    if realism.get("enabled") is not True or realism.get("full_fill_required") is not True:
        raise ValueError("freeze requires execution realism/full-fill policy")
    if realism.get("return_cap_applied") is not False:
        raise ValueError("freeze refuses factory reports with a return cap")
    if realism.get("full_horizon_maturity_required") is not True:
        raise ValueError("freeze requires full-horizon target maturity")
    if realism.get("maturity_fence") != freeze.MATURITY_FENCE:
        raise ValueError("freeze requires the G002m maturity fence")

    if qualification.get("schema_version") != freeze.QUALIFICATION_SCHEMA:
        raise ValueError("freeze requires maturity-safe finalist qualification")
    if qualification.get("generation_id") != generation_id:
        raise ValueError("qualification generation_id mismatch")
    if qualification.get("all_finalists_exactly_reproduced") is not True:
        raise ValueError("not all finalists reproduced exactly")

    qualified_realism = qualification.get("execution_realism") or {}
    for key in (
        "full_fill_required",
        "full_horizon_maturity_required",
        "invalid_target_count",
        "market_quality_manifest",
        "maturity_fence",
        "max_entry_day_participation_pct",
        "max_trailing_adv_participation_pct",
        "return_cap_applied",
        "verified_quality_exclusion_count",
    ):
        if qualified_realism.get(key) != realism.get(key):
            raise ValueError(f"qualification execution realism differs for {key}")


def main() -> None:
    # ``freeze_maturity_safe_finalists`` resolves this module global at runtime,
    # so the compatibility check is scoped to this CLI process and the rest of
    # the already-tested freeze implementation remains unchanged.
    freeze._validate_sources = _validate_sources_compatible
    freeze.main()


if __name__ == "__main__":
    main()
