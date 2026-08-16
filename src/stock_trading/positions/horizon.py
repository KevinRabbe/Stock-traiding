from __future__ import annotations

from datetime import timedelta
from hashlib import sha256

from stock_trading.engine import (
    FeatureSnapshot,
    Opportunity,
    OrderIntent,
    OrderSide,
    PortfolioSnapshot,
)
from stock_trading.market import DuckDbMarketStore


class FixedHorizonPositionManager:
    """Exit positions after their strategy-selected number of observed sessions.

    The manager only inspects bars strictly available by `as_of`. It deliberately
    ignores future bars even when a historical market database contains them.
    Repeat/new candidate information is accepted by the interface for future
    thesis-aware managers but does not change this deterministic baseline.
    """

    def __init__(
        self,
        market_store: DuckDbMarketStore,
        *,
        max_session_lookback: int = 512,
    ) -> None:
        if max_session_lookback <= 0:
            raise ValueError("max_session_lookback must be > 0")
        self.market_store = market_store
        self.max_session_lookback = max_session_lookback

    def orders(
        self,
        portfolio: PortfolioSnapshot,
        as_of,
        candidates: tuple[FeatureSnapshot, ...],
        opportunities: tuple[Opportunity, ...],
    ) -> tuple[OrderIntent, ...]:
        del candidates, opportunities
        orders: list[OrderIntent] = []
        cutoff = as_of.date() + timedelta(days=1)

        for position in sorted(portfolio.positions, key=lambda item: item.position_id):
            raw_horizon = position.metadata.get("horizon_sessions")
            if raw_horizon is None:
                continue
            try:
                horizon = int(raw_horizon)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid horizon_sessions on position {position.position_id}"
                ) from exc
            if horizon <= 0:
                raise ValueError(
                    f"invalid horizon_sessions on position {position.position_id}"
                )
            if horizon > self.max_session_lookback:
                raise ValueError(
                    "position horizon exceeds configured PIT session lookback"
                )

            bars = self.market_store.bars_before(
                position.security_id,
                cutoff,
                self.max_session_lookback,
            )
            held_sessions = sum(bar.date >= position.opened_on for bar in bars)
            if held_sessions < horizon:
                continue

            digest = sha256(
                f"{position.position_id}|fixed-horizon-exit".encode("utf-8")
            ).hexdigest()[:20]
            orders.append(
                OrderIntent(
                    order_id=f"ord_{digest}",
                    strategy_id=position.strategy_id,
                    company_id=position.company_id,
                    security_id=position.security_id,
                    side=OrderSide.SELL,
                    allocation_pct=position.allocation_pct,
                    created_at=as_of,
                    horizon_sessions=horizon,
                    execute_on=as_of.date(),
                    reason="strategy_horizon_complete",
                    metadata={
                        "position_id": position.position_id,
                        "held_sessions": held_sessions,
                    },
                )
            )
        return tuple(orders)
