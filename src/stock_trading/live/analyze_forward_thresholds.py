from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


_DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
_EPSILON = 1e-12


def analyze_forward_rank_thresholds(
    *,
    runtime_dir: str | Path = "data/runtime",
    thresholds: Iterable[float] = _DEFAULT_THRESHOLDS,
    evidence_source: str | None = None,
) -> dict:
    """Evaluate hypothetical rank cutoffs from immutable forward decisions.

    This is diagnostic evidence only. It does not re-score a model, mutate runtime
    state, submit orders, or simulate portfolio constraints. A decision enters the
    threshold cohort only when the original strategy found an economically eligible
    ``chosen_horizon``. The hypothetical threshold then changes only the final rank
    gate, and realized performance is measured at that originally chosen horizon.

    ``evidence_source`` optionally restricts the analysis to a tagged diagnostic
    source such as ``lda_shadow``. Historical diagnostics without an explicit tag are
    treated as ``sec_form4`` for backward-compatible provenance.
    """

    runtime_root = Path(runtime_dir)
    scorecard_path = runtime_root / "forward_scorecard.json"
    payload = _load_scorecard(scorecard_path)
    resolved_thresholds = _normalize_thresholds(thresholds)
    resolved_source = evidence_source.strip() if evidence_source is not None else None
    if evidence_source is not None and not resolved_source:
        raise ValueError("evidence_source must not be empty when provided")

    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("forward scorecard observations must be a list")

    by_strategy: dict[str, list[tuple[dict, dict | None]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    included_observation_count = 0
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("forward scorecard observation must be an object")
        source = _observation_evidence_source(runtime_root, observation)
        if resolved_source is not None and source != resolved_source:
            continue
        source_counts[source] += 1
        included_observation_count += 1
        labels = observation.get("realized_labels")
        decisions = observation.get("strategy_decisions")
        if not isinstance(labels, dict) or not isinstance(decisions, list):
            raise ValueError("forward scorecard observation is incomplete")
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError("forward strategy decision must be an object")
            strategy_id = str(decision.get("strategy_id") or "")
            if not strategy_id:
                raise ValueError("forward strategy decision has no strategy_id")
            chosen_horizon = decision.get("chosen_horizon")
            label = None
            if chosen_horizon is not None:
                try:
                    horizon = int(chosen_horizon)
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid chosen_horizon in forward scorecard") from exc
                raw_label = labels.get(str(horizon))
                if raw_label is not None:
                    if not isinstance(raw_label, dict):
                        raise ValueError("forward realized label must be an object")
                    label = raw_label
            by_strategy[strategy_id].append((decision, label))

    strategies = [
        _strategy_evidence(strategy_id, values, resolved_thresholds)
        for strategy_id, values in sorted(by_strategy.items())
    ]
    matured_threshold_points = sum(
        1
        for strategy in strategies
        for row in strategy["thresholds"]
        if row["matured_selected_decision_count"] > 0
    )
    return {
        "status": "completed",
        "scorecard_path": str(scorecard_path),
        "scorecard_generated_at": payload.get("generated_at"),
        "last_completed_xnys_session": payload.get("last_completed_xnys_session"),
        "interpretation": "diagnostic_only_not_portfolio_simulation",
        "threshold_grid": list(resolved_thresholds),
        "evidence_source_filter": resolved_source,
        "evidence_source_counts": dict(sorted(source_counts.items())),
        "included_observation_count": included_observation_count,
        "strategy_count": len(strategies),
        "matured_threshold_point_count": matured_threshold_points,
        "evidence_ready": matured_threshold_points > 0,
        "strategies": strategies,
    }


def _strategy_evidence(
    strategy_id: str,
    values: list[tuple[dict, dict | None]],
    thresholds: tuple[float, ...],
) -> dict:
    eligible = [
        (decision, label)
        for decision, label in values
        if decision.get("chosen_horizon") is not None
    ]
    observed_rank_thresholds = sorted(
        {
            float(decision.get("rank_threshold", 0.0))
            for decision, _ in values
        }
    )
    rows = [
        _threshold_row(eligible, threshold)
        for threshold in thresholds
    ]
    return {
        "strategy_id": strategy_id,
        "decision_count": len(values),
        "current_emitted_decision_count": sum(
            1 for decision, _ in values if bool(decision.get("emitted"))
        ),
        "economically_eligible_decision_count": len(eligible),
        "no_eligible_horizon_decision_count": len(values) - len(eligible),
        "matured_economically_eligible_decision_count": sum(
            1 for _, label in eligible if label is not None
        ),
        "observed_rank_thresholds": observed_rank_thresholds,
        "thresholds": rows,
    }


def _threshold_row(
    eligible: list[tuple[dict, dict | None]],
    threshold: float,
) -> dict:
    selected = [
        (decision, label)
        for decision, label in eligible
        if float(decision.get("final_percentile", 0.0)) + _EPSILON >= threshold
    ]
    matured = [
        (decision, label)
        for decision, label in selected
        if label is not None
    ]
    labels = [label for _, label in matured if label is not None]
    return {
        "rank_threshold": threshold,
        "selected_decision_count": len(selected),
        "matured_selected_decision_count": len(matured),
        "pending_selected_decision_count": len(selected) - len(matured),
        "average_stock_return": _average(labels, "stock_return"),
        "average_alpha": _average(labels, "alpha"),
        "positive_stock_return_rate": _positive_rate(labels, "stock_return"),
        "positive_alpha_rate": _positive_rate(labels, "alpha"),
        "average_max_favorable_excursion": _average(
            labels,
            "max_favorable_excursion",
        ),
        "average_max_adverse_excursion": _average(
            labels,
            "max_adverse_excursion",
        ),
    }


def _observation_evidence_source(runtime_root: Path, observation: dict) -> str:
    batch_id = str(observation.get("batch_id") or "")
    if not batch_id.startswith("batch_"):
        raise ValueError("forward scorecard observation has invalid batch_id")
    path = runtime_root / "decision_diagnostics" / f"{batch_id}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"decision diagnostic backing forward observation is missing: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid decision diagnostic provenance file: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported decision diagnostic provenance file: {path}")
    if str(payload.get("batch_id") or "") != batch_id:
        raise ValueError(f"decision diagnostic provenance batch mismatch: {path}")
    source = str(payload.get("evidence_source") or "sec_form4").strip()
    if not source:
        raise ValueError(f"decision diagnostic evidence_source is empty: {path}")
    return source


def _average(labels: list[dict], key: str) -> float | None:
    if not labels:
        return None
    try:
        return sum(float(label[key]) for label in labels) / len(labels)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid realized label field: {key}") from exc


def _positive_rate(labels: list[dict], key: str) -> float | None:
    if not labels:
        return None
    try:
        positive = sum(1 for label in labels if float(label[key]) > 0.0)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid realized label field: {key}") from exc
    return positive / len(labels)


def _normalize_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    resolved = tuple(sorted({float(value) for value in values}))
    if not resolved:
        raise ValueError("at least one rank threshold is required")
    if any(value < 0.0 or value > 1.0 for value in resolved):
        raise ValueError("rank thresholds must be between 0 and 1")
    return resolved


def _load_scorecard(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"forward scorecard does not exist yet: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid forward scorecard: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported forward scorecard schema")
    return payload


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        return _normalize_thresholds(
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate hypothetical final-rank cutoffs from immutable forward "
            "decision diagnostics and matured labels."
        )
    )
    parser.add_argument("--runtime-dir", default="data/runtime")
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_DEFAULT_THRESHOLDS,
        help="comma-separated rank thresholds (default: 0.50,0.60,0.70,0.80,0.90,0.95)",
    )
    parser.add_argument(
        "--evidence-source",
        help="optional diagnostic provenance filter, e.g. sec_form4 or lda_shadow",
    )
    args = parser.parse_args()
    result = analyze_forward_rank_thresholds(
        runtime_dir=args.runtime_dir,
        thresholds=args.thresholds,
        evidence_source=args.evidence_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()
