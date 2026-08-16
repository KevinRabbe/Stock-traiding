# Trading engine architecture

The project is no longer organized around extending `lightgbm_profit_vN` indefinitely. Strategy logic is replaceable; capital, risk, positions, execution, lifecycle and monitoring belong to a shared engine.

## Core rule

A strategy may **score opportunities**, but it never owns capital, positions, broker access, or deployment.

```text
point-in-time data
    -> candidate snapshots
    -> strategy plugin
    -> opportunity-level risk
    -> portfolio allocation
    -> portfolio-level risk
    -> position management
    -> broker-neutral order intents
    -> queued/filled execution
    -> persistent portfolio state
    -> audit + monitoring
```

This lets the system use whichever strategy is actually profitable without rewriting the trading stack.

## Research and production share strategy code

```text
RESEARCH
historical PIT candidates + hidden realized outcomes
    -> same OpportunityStrategy plugin
    -> shared risk/portfolio contracts
    -> generic historical backtester
    -> scorecard
    -> challenger registry

RUNTIME
live/paper PIT candidates
    -> same OpportunityStrategy plugin
    -> shared risk/portfolio contracts
    -> position manager
    -> paper/live broker adapter
    -> durable state + audit
```

Realized future outcomes are attached only by the research runner. A strategy receives `FeatureSnapshot` values in both paths and therefore cannot depend on a research-only label interface.

## Champion / challenger lifecycle

Every strategy has a stable `strategy_id` and one stage:

- `development`
- `shadow`
- `paper`
- `live`
- `retired`

Forward promotion is staged:

```text
development -> shadow -> paper -> live
```

Each step has a configurable `ProfitabilityGate`. Paper/live promotion can require an immutable artifact reference. Safety downgrade from live to paper remains available without a profitability gate.

**Stage promotion never changes the champion automatically.** Champion selection is a separate explicit action. A backtest or shadow result can recommend a challenger, but it cannot gain capital authority by itself.

The persistent registry survives restarts and retains metadata for challengers even when their plugins are not currently loaded. A persisted champion cannot execute until its exact plugin is loaded.

## Immutable strategy artifacts

Paper/live strategy artifacts can be represented by `StrategyArtifactManifest`:

- deterministic list of artifact-relative files
- file sizes
- SHA-256 per file
- deterministic manifest SHA-256

Verification fails if a model/config file changes. `StrategyRecord.artifact_ref` can point to the stored manifest rather than a mutable model directory name.

## Stable contracts

### `FeatureSnapshot`

One candidate at one point in time. It carries canonical company/security identity, decision/execution time, and strategy-readable features.

### `Opportunity`

The strategy output contract:

- company/security/event identity
- score/rank
- expected return
- expected alpha
- expected downside
- probability of a positive outcome
- selected holding horizon
- optional strategy-specific metadata

The runtime rejects invented candidates or changed company/security/event identity.

### Opportunity risk

Universal candidate eligibility before portfolio capacity is consumed. Typical rules include maximum predicted downside or minimum expected return after costs.

### Portfolio policy

Chooses how eligible opportunities compete for capital. The baseline is fixed allocation, one active position per company and bounded positions. Sector/peer/correlation optimization can be plugged in later without touching strategy code.

### Portfolio risk

Runs after proposed allocations and may reduce or reject them. It may not increase an allocation. This keeps risk code from becoming a hidden leverage engine.

### Position manager

Existing positions are managed independently from new-entry allocation. The position manager sees current candidates and current strategy opportunities, allowing repeat/new signals to trigger thesis review, but it is restricted to sell/reduction orders against already-open positions.

`FixedHorizonPositionManager` provides the deterministic baseline: it exits after the strategy-selected number of **observed** market sessions and only reads bars available by the current cycle. Future thesis-aware managers can replace it.

### Execution broker

`OrderIntent` is broker-neutral and contains an optional intended execution date. Execution status is explicit:

- `queued`
- `filled`
- `rejected`
- `cancelled`

The engine settles previously queued orders before taking the next portfolio snapshot. This permits a strategy decision today to schedule an order for the next market session without pretending it filled immediately.

### Prepared engine cycle

`PreparedEngineCycle` captures one immutable post-settlement portfolio + PIT candidate view. The champion and shadow challengers can evaluate the exact same context without duplicate ingestion or different portfolio state.

## Shadow challengers

`ShadowStrategyEvaluator` runs loaded strategies in the `shadow` stage against the champion's prepared context.

It applies the same opportunity-risk, portfolio-allocation and portfolio-risk policies to compute realistic **would-trade** selections, but it never creates orders or calls a broker.

`TradingService` therefore runs:

```text
settle pending orders
    -> capture portfolio + candidates once
    -> evaluate shadow challengers with no broker authority
    -> run champion on the same prepared context
    -> execute champion orders only
```

Stateful strategies such as the V5 rolling-calibration adapter advance their own state once per service cycle.

## Persistent paper execution

The included paper execution layer provides:

- atomic durable cash/position state
- durable pending orders
- completed execution reports
- idempotency by order ID across restart/retry
- future-date order queuing
- no stale-price fallback for due orders
- per-side transaction-cost modeling
- duplicate-company safety rejection
- partial/full sells
- mark-to-market `PortfolioSnapshot`

The first DuckDB price adapter uses a same-date daily bar only. A richer intraday/realtime provider can replace the price adapter without changing strategy or engine contracts.

## Auditing

Champion/challenger metadata is atomically persisted. Completed engine cycles can be written to an append-only, fsync-backed JSONL audit journal. Every order/execution/state transition therefore has a durable route into monitoring.

## Current package direction

```text
stock_trading/
    existing source + feature modules/

    engine/
        artifacts.py
        contracts.py
        lifecycle.py
        persistence.py
        policies.py
        protocols.py
        registry.py
        runtime.py

    strategies/
        v5_adaptive_horizon.py
        future_strategy.py
        ensemble.py

    research/
        historical.py
        walk_forward.py

    positions/
        horizon.py
        future_thesis_manager.py

    execution/
        paper.py
        prices.py
        future_live_broker.py

    live/
        service.py

    experiments/
        strategy_engine_v5_replay.py
        legacy V1-V7 experiments remain for reproducibility
```

## Architecture migration status

1. **Engine contracts / authority boundaries** — implemented.
2. **V5 strategy adapter** — implemented; saved 5/20/60 models can run from generic feature snapshots.
3. **Unified historical strategy runner** — implemented.
4. **Exact V5 architecture replay guard** — implemented; local run must reproduce V5 year-by-year before V5 is registered as the first paper champion.
5. **Persistent strategy registry + audit** — implemented.
6. **Persistent queued paper execution + portfolio state** — implemented.
7. **PIT fixed-horizon baseline position management** — implemented.
8. **Prepared-cycle shadow/champion runtime** — implemented.
9. **Controlled strategy lifecycle/promotion** — implemented.
10. **Immutable artifact manifests** — implemented.

After exact local V5 replay passes, the core architecture is ready for strategy work again. New approaches should be implemented as `OpportunityStrategy` plugins or shared portfolio/position modules—not as another copy of the trading engine.

## What remains adapter-specific

These are integrations, not missing core architecture:

- live PIT candidate-source assembly from the existing event/feature pipeline
- whichever real broker API is selected later
- realtime/intraday price provider if needed
- UI/dashboard/notifications
- richer sector/peer/correlation portfolio modules
- thesis-aware position manager

Those pieces can be added independently without changing the strategy contract.

## What happens to V1-V7?

Nothing is deleted. They remain reproducible research history.

V5 is the first strategy adapter because it is the current strongest development baseline. V6/V7 remain evidence about capital allocation, but their sizing formulas are not embedded into the production architecture.

Future candidates can be LightGBM variants, sector/peer-aware models, ensembles, different targets or completely different model families. They compete through the same historical runner, scorecard, shadow/paper lifecycle and explicit champion promotion.

## Non-negotiable integrity rules

- point-in-time data only
- no label/future-bar leakage
- canonical company/security identity
- transaction costs in research and paper execution
- deterministic strategy/model artifacts
- strategy cannot submit broker orders directly
- position management cannot secretly increase exposure
- portfolio risk cannot silently upsize allocation
- backtest/shadow results cannot silently promote a champion
- paper/live execution share the same runtime contracts
- queued orders settle before the next portfolio snapshot
- every order/execution/deployment transition is auditable
