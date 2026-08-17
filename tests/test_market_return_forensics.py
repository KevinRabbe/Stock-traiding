from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_trading.experiments.market_return_forensics import audit_target
from stock_trading.market.models import MarketBar


def _bar(
    day: int,
    *,
    raw_open: str,
    raw_close: str,
    adj_open: str,
    adj_close: str,
    split_factor: str = "1",
) -> MarketBar:
    raw_o = Decimal(raw_open)
    raw_c = Decimal(raw_close)
    adj_o = Decimal(adj_open)
    adj_c = Decimal(adj_close)
    return MarketBar(
        security_id="sec-1",
        ticker="XYZ",
        date=date(2020, 1, day),
        open=raw_o,
        high=max(raw_o, raw_c),
        low=min(raw_o, raw_c),
        close=raw_c,
        volume=Decimal("1000"),
        adj_open=adj_o,
        adj_high=max(adj_o, adj_c),
        adj_low=min(adj_o, adj_c),
        adj_close=adj_c,
        adj_volume=Decimal("1000"),
        dividend_cash=Decimal("0"),
        split_factor=Decimal(split_factor),
    )


def test_audit_target_exposes_adjustment_and_split_discontinuities() -> None:
    bars = (
        _bar(2, raw_open="10", raw_close="11", adj_open="5", adj_close="5.5"),
        _bar(
            3,
            raw_open="5.5",
            raw_close="6",
            adj_open="5.5",
            adj_close="6",
            split_factor="2",
        ),
        _bar(4, raw_open="6", raw_close="12", adj_open="6", adj_close="12"),
    )

    finding = audit_target(
        event_id="event-1",
        company_id="company-1",
        security_id="sec-1",
        horizon=3,
        expected_adjusted_return=1.4,
        entry=bars[0],
        exit_bar=bars[-1],
        series=bars,
    )

    assert finding.adjusted_return == pytest.approx(1.4)
    assert finding.raw_return == pytest.approx(0.2)
    assert finding.entry_adjustment_factor == pytest.approx(0.5)
    assert finding.exit_adjustment_factor == pytest.approx(1.0)
    assert finding.adjustment_factor_ratio == pytest.approx(2.0)
    assert finding.non_unit_split_factors == (
        {
            "date": "2020-01-03",
            "split_factor": 2.0,
            "raw_close": 6.0,
            "adj_close": 6.0,
        },
    )
    assert finding.largest_adjusted_close_jump["return"] == pytest.approx(1.0)
    assert finding.largest_raw_close_jump["return"] == pytest.approx(1.0)


def test_audit_target_fails_when_reconstructed_bars_disagree_with_target() -> None:
    bars = (
        _bar(2, raw_open="10", raw_close="10", adj_open="10", adj_close="10"),
        _bar(3, raw_open="10", raw_close="11", adj_open="10", adj_close="11"),
    )

    with pytest.raises(ValueError, match="target/bar mismatch"):
        audit_target(
            event_id="event-1",
            company_id="company-1",
            security_id="sec-1",
            horizon=2,
            expected_adjusted_return=0.5,
            entry=bars[0],
            exit_bar=bars[-1],
            series=bars,
        )
