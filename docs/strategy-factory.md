# Strategy factory

The strategy factory exploits the fact that the project's current LightGBM models are cheap to train. Instead of treating one model as the trading system, a generation trains a reproducible population of structurally different small strategies from scratch and evaluates every one through the same strategy-agnostic historical engine.

## Core principle

```text
fresh historical PIT data
    -> constrained strategy population
    -> parallel from-scratch training
    -> unified walk-forward backtest
    -> profitability gate
    -> trade-overlap/diversity filter
    -> finalists
    -> retrain exact finalist spec
    -> immutable artifact
    -> shadow -> paper -> explicit champion promotion
```

Screening models are intentionally not persisted. The exact variant specification and results are persisted, because retraining is cheaper and cleaner than carrying hundreds of model artifacts forever.

## Generation design space

The first factory varies named structural choices rather than running a continuous hyperparameter optimizer:

- feature view: full, market/regime, event/history, balanced core
- training history: expanding, trailing 5 years, trailing 8 years
- tree profile: conservative, baseline, expressive
- horizon set: 5/20/60, 5/20, 20/60 sessions
- alpha contribution to rank: 0%, 25%, 50%
- two deterministic LightGBM seeds

This creates a larger design grid than one generation tests. A generation deterministically samples that grid from its generation seed, and the report records the complete hypothesis count and every attempted specification. That makes search breadth explicit instead of hiding it behind the winning result.

## Selection

A strategy first has to clear simple economic gates:

- positive compounded return
- minimum profit factor
- minimum trade count
- maximum realized drawdown

Eligible strategies are ranked on a transparent composite of:

- compounded return: 30%
- aggregate profit factor: 20%
- profitable-year rate: 15%
- low drawdown: 15%
- compounded return excluding the best year: 20%

The factory then greedily rejects finalists whose historical opportunity set has excessive Jaccard overlap with an already-selected finalist. The objective is not merely to find eight versions of the same strategy.

A lower-return specialist can therefore survive when it contributes genuinely different trades.

## Parallelism

The CLI parallelizes across strategy variants using process workers. Each worker loads the local PIT training rows and market targets once, then trains all annual walk-forward models for the variants assigned to it. `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are bounded before worker creation so the outer strategy population, rather than each individual LightGBM, is the main parallelism axis.

## First generation

After the V5 architecture replay succeeds, a reasonable first run is:

```powershell
.\.venv\Scripts\python.exe -m stock_trading.experiments.lightgbm_strategy_factory `
  --experiment-dir data\experiments\lightgbm_holdout_250_v2 `
  --market-db data\normalized\market.duckdb `
  --benchmark-security-id benchmark_spy `
  --generation-id g001 `
  --population-size 48 `
  --workers 4 `
  --threads-per-worker 2
```

The report is written to:

```text
data/experiments/lightgbm_holdout_250_v2/strategy_factory/g001/report.json
```

No strategy is promoted automatically. The report identifies finalists; a later finalist-freezing step retrains exact finalist specs and creates immutable artifacts for shadow evaluation.

## What the factory deliberately does not do

- It does not optimize directly on the future test year.
- It does not expose realized outcomes to the strategy plugin.
- It does not save every screening model.
- It does not auto-promote the highest-return backtest.
- It does not treat random seeds as independent economic strategies when their trades are nearly identical.
- It does not search arbitrary continuous hyperparameters until something looks good.

The factory exists to turn cheap model training into breadth of hypotheses while keeping the capital/deployment path conservative.
