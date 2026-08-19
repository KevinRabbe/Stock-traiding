from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from stock_trading.core import as_utc
from stock_trading.market.execution_time import decision_market_date

from .forward_outcomes import refresh_forward_outcome_scorecard
from .poll_sec_form4 import poll_current_form4
from .run_current_paper_shadow import run_current_paper_shadow_cycle
from .session_calendar import XnysExecutionSessionResolver


@dataclass(frozen=True, slots=True)
class PipelineSessionGuard:
    allowed: bool
    reason: str | None
    polled_target_execution_date: date
    current_target_execution_date: date
    seconds_until_target_open: float | None


def evaluate_pipeline_session_guard(
    resolver: XnysExecutionSessionResolver,
    *,
    polled_target_execution_date: date,
    now: datetime,
    minimum_preopen_buffer: timedelta = timedelta(minutes=15),
) -> PipelineSessionGuard:
    """Fail closed if SEC polling consumed an execution-session boundary.

    A poll can take long enough to cross the NYSE open. The pending selection was
    classified at poll start, so the orchestration layer must re-evaluate the
    executable session before giving the PAPER runner any authority. For a same-day
    pre-open target we additionally require a safety buffer, preventing a cycle from
    starting seconds before the opening bell and finishing after the opportunity was
    already missed.
    """

    cutoff = as_utc(now)
    current_target = resolver.cycle_execution_date(cutoff)
    if current_target != polled_target_execution_date:
        return PipelineSessionGuard(
            allowed=False,
            reason="session_boundary_crossed_after_poll",
            polled_target_execution_date=polled_target_execution_date,
            current_target_execution_date=current_target,
            seconds_until_target_open=None,
        )

    eastern_day = decision_market_date(cutoff)
    if current_target == eastern_day:
        seconds_until_open = (resolver.session_open(current_target) - cutoff).total_seconds()
        if seconds_until_open < minimum_preopen_buffer.total_seconds():
            return PipelineSessionGuard(
                allowed=False,
                reason="preopen_safety_buffer_too_short",
                polled_target_execution_date=polled_target_execution_date,
                current_target_execution_date=current_target,
                seconds_until_target_open=seconds_until_open,
            )
        return PipelineSessionGuard(
            allowed=True,
            reason=None,
            polled_target_execution_date=polled_target_execution_date,
            current_target_execution_date=current_target,
            seconds_until_target_open=seconds_until_open,
        )

    return PipelineSessionGuard(
        allowed=True,
        reason=None,
        polled_target_execution_date=polled_target_execution_date,
        current_target_execution_date=current_target,
        seconds_until_target_open=None,
    )


def run_current_pipeline(
    *,
    data_root: str | Path = "data",
    experiment_dir: str | Path = "data/experiments/lightgbm_holdout_250_v2",
    runtime_dir: str | Path = "data/runtime",
    initial_lookback_days: int = 7,
    max_companies: int | None = None,
    minimum_preopen_buffer_minutes: int = 15,
) -> dict:
    """Poll current Form 4 intake, guard the session boundary, then run PAPER+SHADOW.

    This is the preferred operational entry point. SEC intake remains non-authority;
    PAPER authority begins only after polling completes, the current XNYS target is
    re-resolved, and any same-day target retains the configured pre-open buffer.

    After the authoritative cycle is durable, a separate measurement-only step
    refreshes realized labels for prior forward decision candidates. Outcome refresh
    can never acknowledge events, place orders, or mutate strategy calibration.
    """

    if minimum_preopen_buffer_minutes <= 0:
        raise ValueError("minimum_preopen_buffer_minutes must be > 0")

    poll_started_at = datetime.now(timezone.utc)
    poll_payload = poll_current_form4(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        initial_lookback_days=initial_lookback_days,
        max_companies=max_companies,
        as_of=poll_started_at,
    )
    after_poll = datetime.now(timezone.utc)

    selection = poll_payload.get("pending_session_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("current Form 4 poll did not return pending-session diagnostics")
    try:
        polled_target = date.fromisoformat(str(selection["target_execution_date"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("current Form 4 poll returned an invalid target session") from exc

    resolver = XnysExecutionSessionResolver()
    guard = evaluate_pipeline_session_guard(
        resolver,
        polled_target_execution_date=polled_target,
        now=after_poll,
        minimum_preopen_buffer=timedelta(minutes=minimum_preopen_buffer_minutes),
    )
    guard_payload = {
        "allowed": guard.allowed,
        "reason": guard.reason,
        "polled_target_execution_date": guard.polled_target_execution_date.isoformat(),
        "current_target_execution_date": guard.current_target_execution_date.isoformat(),
        "seconds_until_target_open": guard.seconds_until_target_open,
        "minimum_preopen_buffer_minutes": minimum_preopen_buffer_minutes,
    }
    if not guard.allowed:
        return {
            "status": guard.reason,
            "poll_started_at": poll_started_at.isoformat(),
            "post_poll_as_of": after_poll.isoformat(),
            "session_guard": guard_payload,
            "poll": poll_payload,
            "cycle": None,
            "forward_outcomes": None,
        }

    # Use a fresh post-poll timestamp. Never carry the poll-start timestamp into the
    # trading cycle; doing so could make an already-crossed wall-clock boundary look
    # artificially actionable.
    cycle_payload = run_current_paper_shadow_cycle(
        data_root=data_root,
        runtime_dir=runtime_dir,
        as_of=after_poll,
    )

    # The trading cycle above is already authoritative/durable. Forward outcome
    # refresh is intentionally downstream measurement. Return a structured error
    # instead of turning a completed PAPER receipt into an ambiguous traceback.
    try:
        forward_outcomes = refresh_forward_outcome_scorecard(
            data_root=data_root,
            runtime_dir=runtime_dir,
            as_of=datetime.now(timezone.utc),
        )
    except Exception as exc:  # measurement must not rewrite trading semantics
        forward_outcomes = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    cycle_status = str(cycle_payload.get("status", "unknown"))
    status = (
        cycle_status
        if forward_outcomes.get("status") == "completed"
        else f"{cycle_status}_with_forward_outcome_refresh_error"
    )
    return {
        "status": status,
        "poll_started_at": poll_started_at.isoformat(),
        "post_poll_as_of": after_poll.isoformat(),
        "session_guard": guard_payload,
        "poll": poll_payload,
        "cycle": cycle_payload,
        "forward_outcomes": forward_outcomes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Poll current SEC Form 4 filings and, only after re-validating the XNYS "
            "execution window, run one persisted PAPER champion + SHADOW cycle and "
            "refresh realized forward-decision outcomes."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("data/experiments/lightgbm_holdout_250_v2"),
    )
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--initial-lookback-days", type=int, default=7)
    parser.add_argument("--max-companies", type=int)
    parser.add_argument("--minimum-preopen-buffer-minutes", type=int, default=15)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_current_pipeline(
        data_root=args.data_root,
        experiment_dir=args.experiment_dir,
        runtime_dir=args.runtime_dir,
        initial_lookback_days=args.initial_lookback_days,
        max_companies=args.max_companies,
        minimum_preopen_buffer_minutes=args.minimum_preopen_buffer_minutes,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
