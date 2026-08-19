from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from stock_trading.core import as_utc
from stock_trading.execution import (
    DuckDbLatestClosePriceProvider,
    FilePaperLedger,
    PaperPortfolioStateProvider,
    SessionBarPaperExecutionBroker,
)
from stock_trading.local_secrets import load_tiingo_credentials
from stock_trading.market import DuckDbMarketStore, TiingoClient, TiingoNormalizer
from stock_trading.positions import FixedHorizonPositionManager
from stock_trading.storage import FileRawStore

from .run_current_paper_shadow import _load_runtime_config
from .session_calendar import XnysExecutionSessionResolver


@dataclass(frozen=True, slots=True)
class PaperLifecycleMarketSync:
    tracked_security_count: int
    downloaded_series_count: int
    bars_added: int
    lagging_series: tuple[str, ...]


def service_current_paper_lifecycle(
    *,
    data_root: str | Path = "data",
    runtime_dir: str | Path = "data/runtime",
    as_of: datetime | None = None,
) -> dict:
    """Service durable PAPER orders/positions without requiring a new signal batch.

    This authority is deliberately independent from SEC intake. On every operational
    pipeline run it extends market data for securities already present in the PAPER
    ledger through the last completed XNYS session, settles due queued orders against
    their original execution session, and emits exact-horizon full exits. Re-running
    the same completed session is idempotent because order IDs and the PAPER ledger
    are durable.
    """

    data_root = Path(data_root)
    runtime_dir = Path(runtime_dir)
    cutoff = as_utc(as_of or datetime.now(timezone.utc))
    config = _load_runtime_config(runtime_dir)
    resolver = XnysExecutionSessionResolver()
    completed_session = resolver.last_completed_session(cutoff)
    lifecycle_at = resolver.session_close(completed_session)

    market_store = DuckDbMarketStore(Path(str(config["market_db"])))
    market_store.enable_read_cache(max_series=32)
    ledger = FilePaperLedger(Path(str(config["paper_ledger"])), starting_cash=10_000.0)
    initial_state = ledger.load()
    security_ids = tuple(
        sorted(
            {
                *(item.security_id for item in initial_state.positions),
                *(item.security_id for item in initial_state.pending_orders),
            }
        )
    )

    market_sync = _sync_lifecycle_market(
        security_ids,
        data_root=data_root,
        market_store=market_store,
        completed_session=completed_session,
    )
    market_store.clear_read_cache()

    broker = SessionBarPaperExecutionBroker(
        ledger,
        market_store,
        per_side_cost_bps=10.0,
    )
    settled = broker.settle(lifecycle_at)

    mark_provider = DuckDbLatestClosePriceProvider(market_store)
    snapshot = PaperPortfolioStateProvider(ledger, mark_provider).snapshot(lifecycle_at)
    exit_orders = FixedHorizonPositionManager(market_store).orders(
        snapshot,
        lifecycle_at,
        (),
        (),
    )
    exit_reports = broker.execute(exit_orders)
    final_state = ledger.load()

    return {
        "status": "completed",
        "as_of": cutoff.isoformat(),
        "completed_session": completed_session.isoformat(),
        "lifecycle_at": lifecycle_at.isoformat(),
        "market_sync": {
            "tracked_security_count": market_sync.tracked_security_count,
            "downloaded_series_count": market_sync.downloaded_series_count,
            "bars_added": market_sync.bars_added,
            "lagging_series_count": len(market_sync.lagging_series),
            "lagging_series": list(market_sync.lagging_series),
        },
        "settled_order_count": len(settled),
        "settled_orders": [_report_payload(item) for item in settled],
        "generated_exit_order_count": len(exit_orders),
        "generated_exit_order_ids": [item.order_id for item in exit_orders],
        "exit_reports": [_report_payload(item) for item in exit_reports],
        "open_position_count": len(final_state.positions),
        "pending_order_count": len(final_state.pending_orders),
        "completed_report_count": len(final_state.completed_reports),
    }


def _sync_lifecycle_market(
    security_ids: tuple[str, ...],
    *,
    data_root: Path,
    market_store: DuckDbMarketStore,
    completed_session: date,
) -> PaperLifecycleMarketSync:
    if not security_ids:
        return PaperLifecycleMarketSync(0, 0, 0, ())

    targets: list[tuple[str, str, date]] = []
    lagging: list[str] = []
    for security_id in security_ids:
        latest = market_store.bars_before(
            security_id,
            completed_session + timedelta(days=1),
            1,
        )
        if not latest:
            raise RuntimeError(
                f"PAPER ledger security has no stored market history: {security_id}"
            )
        ticker = latest[-1].ticker
        bounds = market_store.date_bounds(security_id, ticker)
        if bounds is None:
            raise RuntimeError(
                f"PAPER ledger security has no ticker bounds: {security_id}/{ticker}"
            )
        if bounds[1] < completed_session:
            targets.append((security_id, ticker, bounds[1] + timedelta(days=1)))

    downloaded = 0
    bars_added = 0
    if targets:
        raw_store = FileRawStore(data_root / "raw")
        credentials = load_tiingo_credentials(data_root)
        with TiingoClient(credentials.token) as client:
            for security_id, ticker, start in targets:
                raw = client.fetch_prices(ticker, start, completed_session)
                raw_store.put(raw)
                bars = TiingoNormalizer().parse_prices(
                    raw,
                    security_id=security_id,
                    ticker=ticker,
                )
                before = market_store.count_bars(
                    security_id,
                    ticker,
                    start,
                    completed_session,
                )
                market_store.put_many(bars)
                after = market_store.count_bars(
                    security_id,
                    ticker,
                    start,
                    completed_session,
                )
                downloaded += 1
                bars_added += max(0, after - before)

                latest_after = market_store.bars_before(
                    security_id,
                    completed_session + timedelta(days=1),
                    1,
                )
                if not latest_after or latest_after[-1].date < completed_session:
                    lagging.append(security_id)

    return PaperLifecycleMarketSync(
        tracked_security_count=len(security_ids),
        downloaded_series_count=downloaded,
        bars_added=bars_added,
        lagging_series=tuple(sorted(lagging)),
    )


def _report_payload(report) -> dict:
    return {
        "order_id": report.order_id,
        "status": report.status.value,
        "accepted": report.accepted,
        "executed_at": report.executed_at.isoformat(),
        "fill_price": report.fill_price,
        "message": report.message,
    }
