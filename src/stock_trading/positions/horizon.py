from __future__ import annotations

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
    """Exit positions at the close of their exact strategy-selected horizon.

    A restart may happen several sessions after the intended exit. The manager
    therefore derives the terminal session from observed bars starting at the
    position's opening session and keeps that original date in ``execute_on``.
    Future rows are ignored even when a historical database contains them.
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

            first_horizon_bars = self.market_store.bars_from(
                position.security_id,
                position.opened_on,
                horizon,
            )
            observed = tuple(
                bar for bar in first_horizon_bars if bar.date <= as_of.date()
            )
            if len(observed) < horizon:
                continue
            exit_date = observed[horizon - 1].date

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
                    execute_on=exit_date,
                    reason="strategy_horizon_complete",
                    metadata={
                        "position_id": position.position_id,
                        "held_sessions": len(observed),
                        "intended_exit_date": exit_date.isoformat(),
                        "full_exit": True,
                    },
                )
            )
        return tuple(orders)
