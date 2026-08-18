from datetime import date, datetime, timedelta, timezone

from stock_trading.live.run_current_pipeline import evaluate_pipeline_session_guard


UTC = timezone.utc


class _Resolver:
    def __init__(self, current_target: date, open_at: datetime) -> None:
        self.current_target = current_target
        self.open_at = open_at

    def cycle_execution_date(self, now: datetime) -> date:
        del now
        return self.current_target

    def session_open(self, session_date: date) -> datetime:
        assert session_date == self.current_target
        return self.open_at


def test_pipeline_guard_rejects_target_change_after_poll() -> None:
    now = datetime(2026, 8, 18, 13, 31, tzinfo=UTC)
    resolver = _Resolver(
        date(2026, 8, 19),
        datetime(2026, 8, 19, 13, 30, tzinfo=UTC),
    )

    result = evaluate_pipeline_session_guard(
        resolver,  # type: ignore[arg-type]
        polled_target_execution_date=date(2026, 8, 18),
        now=now,
    )

    assert result.allowed is False
    assert result.reason == "session_boundary_crossed_after_poll"
    assert result.current_target_execution_date == date(2026, 8, 19)


def test_pipeline_guard_requires_same_day_preopen_buffer() -> None:
    target = date(2026, 8, 18)
    open_at = datetime(2026, 8, 18, 13, 30, tzinfo=UTC)
    resolver = _Resolver(target, open_at)

    too_late = evaluate_pipeline_session_guard(
        resolver,  # type: ignore[arg-type]
        polled_target_execution_date=target,
        now=datetime(2026, 8, 18, 13, 20, tzinfo=UTC),
        minimum_preopen_buffer=timedelta(minutes=15),
    )
    assert too_late.allowed is False
    assert too_late.reason == "preopen_safety_buffer_too_short"
    assert too_late.seconds_until_target_open == 600.0

    early = evaluate_pipeline_session_guard(
        resolver,  # type: ignore[arg-type]
        polled_target_execution_date=target,
        now=datetime(2026, 8, 18, 13, 0, tzinfo=UTC),
        minimum_preopen_buffer=timedelta(minutes=15),
    )
    assert early.allowed is True
    assert early.reason is None
    assert early.seconds_until_target_open == 1800.0


def test_pipeline_guard_allows_future_session_without_preopen_clock_race() -> None:
    target = date(2026, 8, 19)
    resolver = _Resolver(
        target,
        datetime(2026, 8, 19, 13, 30, tzinfo=UTC),
    )

    result = evaluate_pipeline_session_guard(
        resolver,  # type: ignore[arg-type]
        polled_target_execution_date=target,
        now=datetime(2026, 8, 18, 18, 0, tzinfo=UTC),
    )

    assert result.allowed is True
    assert result.reason is None
    assert result.seconds_until_target_open is None
