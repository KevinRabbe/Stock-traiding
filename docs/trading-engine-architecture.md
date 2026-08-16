# Trading engine architecture

This document defines the target architecture for the project after the V1-V7 research phase. The goal is to stop coupling the trading system to one LightGBM experiment and make strategy choice replaceable.

## Core rule

A strategy may **score opportunities**, but it never owns capital, positions, broker access, or live deployment.

The shared engine owns the rest:

```text
point-in-time data
    -> candidate snapshots
    -> active strategy
    -> opportunity-level risk
    -> portfolio allocation
    -> portfolio-level risk
    -> existing-position management
    -> execution broker
    -> state + monitoring
```

This lets the project keep whichever strategy is profitable without rewriting the production stack.

## Research and production are separate

```text
RESEARCH
historical PIT data
    -> strategy plugin
    -> walk-forward/backtest
    -> scorecard
    -> challenger registry
    -> shadow/paper evaluation
    -> explicit promotion

PRODUCTION
live PIT candidate source
    -> approved champion strategy
    -> shared portfolio/risk engine
    -> broker
    -> persistent portfolio/order state
    -> monitoring
```

A profitable backtest may produce a recommendation, but it must never automatically deploy itself live.

## Champion / challenger model

Every strategy has a stable `strategy_id` and a lifecycle stage:

- `development`
- `shadow`
- `paper`
- `live`
- `retired`

The registry stores a scorecard and an externally computed `selection_score`. A configurable profitability gate decides which strategies are eligible for comparison. The registry can recommend a challenger, but champion promotion is explicit.

This means future strategies can be compared on whatever production objective we decide matters most: compounded return, profit factor, drawdown, trade count, consistency, paper performance, or a composite score. The engine itself does not hard-code one research objective.

## Stable contracts

### `FeatureSnapshot`

One candidate at one point in time. It carries canonical company/security identity, decision/execution time, and strategy-readable features.

### `Opportunity`

The only output a strategy needs to provide:

- company/security identity
- score/rank
- expected return
- expected alpha
- expected downside
- probability of a positive outcome
- chosen holding horizon
- optional strategy-specific metadata

The runtime verifies that a strategy cannot mutate candidate identity or invent candidates that were not present in the PIT candidate set.

### Opportunity risk

Runs before allocation. This is where universal eligibility rules belong, such as a maximum predicted downside or minimum expected return after costs.

Rejecting an opportunity here does not consume a portfolio slot.

### Portfolio policy

Chooses how eligible opportunities compete for capital. The current baseline remains fixed allocation, one active position per company, bounded slots and bounded gross exposure.

Future implementations can add sector/peer concentration, correlation, regime exposure, or optimizer-based capital allocation without changing strategy plugins.

### Portfolio risk

Runs after proposed allocations. It may reduce or reject allocations but may not increase them. This boundary prevents a risk module from becoming a hidden leverage engine.

### Position manager

Existing positions are managed separately from new-entry opportunity generation. This is where the final system can implement:

- hold
- exit
- reduce
- extend/shorten horizon
- thesis re-evaluation after a new event

Repeat events therefore become position-state updates instead of automatically opening tranches.

### Execution broker

The engine emits broker-neutral `OrderIntent` objects. Paper, shadow and live brokers implement the same execution protocol.

The broker must return exactly one execution result for every submitted order, which gives monitoring/state code an auditable lifecycle.

## Intended package layout

The current experiment modules remain as research history. Production gradually moves toward:

```text
stock_trading/
    data + existing source modules/
    features/
    engine/
        contracts.py
        protocols.py
        registry.py
        policies.py
        runtime.py
    strategies/
        v5_adaptive_horizon.py
        future_strategy.py
        ensemble.py
    portfolio/
        allocation.py
        exposure.py
        peers.py
    positions/
        manager.py
        thesis.py
    execution/
        paper.py
        live.py
    research/
        runner.py
        scorecards.py
        promotion.py
    live/
        service.py
    monitoring/
        events.py
        reports.py
```

The exact folder split can evolve; the stable engine contracts should not.

## Migration path

1. **Architecture foundation** — contracts, runtime boundaries, registry and safe baseline policies. No model behavior changes.
2. **V5 adapter** — wrap the current saved adaptive 5/20/60-session model as the first strategy plugin and verify identical opportunity/trade output.
3. **Unified research runner** — run any registered strategy through the same backtest/scorecard interface rather than adding `lightgbm_profit_v8`, `v9`, etc.
4. **Persistent strategy/model registry** — artifact hashes, configs, scorecards, stage, champion/challenger history.
5. **Paper execution + persistent portfolio state** — same engine path as live, different broker adapter.
6. **Active position manager** — re-score open positions after new events/regime changes.
7. **Richer portfolio context** — dynamic peers/sector history/correlation/exposure controls.
8. **Live service + monitoring** — scheduled candidate ingestion, deterministic orders, audit logs, health/status dashboard.

## What happens to V1-V7?

Nothing is deleted. They remain reproducible research experiments.

The current adaptive-horizon V5 behavior is the best starting strategy candidate. V6/V7 remain useful evidence about portfolio sizing, but the production architecture no longer needs to inherit either sizing formula. Once V5 is adapted, future improvements become new strategy or portfolio plugins and can be compared on equal footing.

## Non-negotiable integrity rules

- point-in-time data only
- no label/future-bar leakage
- canonical company/security identity
- transaction costs in research
- deterministic strategy/model artifact references
- strategy cannot submit broker orders directly
- risk cannot silently upsize allocation
- backtest results cannot silently promote a live strategy
- paper/live execution share the same runtime path
- every order/execution/state transition is auditable
