from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from stock_trading.core import as_utc
from stock_trading.market.execution_time import decision_market_date

from .forward_outcomes import refresh_forward_outcome_scorecard
from .paper_lifecycle import service_current_paper_lifecycle
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
    """Fail closed if polling/lifecycle work consumed an execution-session boundary.

    A poll or PAPER lifecycle catch-up can take long enough to cross the NYSE open.
    Pending selection was classified at poll start, so the orchestration layer must
    re-evaluate the executable session after both non-signal phases finish. For a
    same-day pre-open target we additionally require a safety buffer.
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
    """Poll SEC, service durable PAPER state, guard the session, then evaluate signals.

    PAPER lifecycle servicing is independent from new SEC events: due queued entries
    and exact-horizon exits are caught up through the last completed XNYS session on
    every invocation. New decision authority begins only after that catch-up and a
    fresh execution-session guard. Forward outcomes remain downstream measurement.
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

    try:
        paper_lifecycle = service_current_paper_lifecycle(
            data_root=data_root,
            runtime_dir=runtime_dir,
            as_of=after_poll,
        )
    except Exception as exc:
        after_lifecycle = datetime.now(timezone.utc)
        return {
            "status": "paper_lifecycle_error",
            "poll_started_at": poll_started_at.isoformat(),
            "post_poll_as_of": after_poll.isoformat(),
            "post_lifecycle_as_of": after_lifecycle.isoformat(),
            "session_guard": None,
            "poll": poll_payload,
            "paper_lifecycle": {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            "cycle": None,
            "forward_outcomes": None,
        }

    after_lifecycle = datetime.now(timezone.utc)
    resolver = XnysExecutionSessionResolver()
    guard = evaluate_pipeline_session_guard(
        resolver,
        polled_target_execution_date=polled_target,
        now=after_lifecycle,
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
            "post_lifecycle_as_of": after_lifecycle.isoformat(),
            "session_guard": guard_payload,
            "poll": poll_payload,
            "paper_lifecycle": paper_lifecycle,
            "cycle": None,
            "forward_outcomes": None,
        }

    # Use a fresh post-lifecycle timestamp. Never carry the poll-start timestamp into
    # the trading cycle; doing so could make a crossed wall-clock boundary actionable.
    cycle_payload = run_current_paper_shadow_cycle(
        data_root=data_root,
        runtime_dir=runtime_dir,
        as_of=after_lifecycle,
    )

    # The trading/lifecycle authority above is already durable. Forward outcome
    # refresh is measurement only and cannot turn a completed PAPER operation into
    # an ambiguous traceback.
    try:
        forward_outcomes = refresh_forward_outcome_scorecard(
            data_root=data_root,
            runtime_dir=runtime_dir,
            as_of=datetime.now(timezone.utc),
        )
    except Exception as exc:
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
        "post_lifecycle_as_of": after_lifecycle.isoformat(),
        "session_guard": guard_payload,
        "poll": poll_payload,
        "paper_lifecycle": paper_lifecycle,
        "cycle": cycle_payload,
        "forward_outcomes": forward_outcomes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Poll current SEC Form 4 filings, catch up durable PAPER orders/positions "
            "through the last completed XNYS session, re-validate the execution window, "
            "run one PAPER+SHADOW decision batch, and refresh forward outcomes."
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
