# Stock-traiding

Private trading-system research and implementation repository.

The project is being built around point-in-time-safe alternative data, semantic extraction, machine-learning opportunity ranking, deterministic risk controls, and deliberately simple execution.

## Implemented foundations

### Milestone 1 — core contracts

- immutable raw records and canonical sparse events
- strict timezone-aware UTC timestamps
- deterministic identifiers for idempotent ingestion
- typed source payloads
- Qwen semantic annotations isolated from authoritative source fields
- minimal collector/normalizer interfaces

### Milestone 2 — SEC insiders

- quarterly Form 3/4/5 archive parsing with V1 normalization of non-derivative Form 4/4-A transactions
- live Form 4 XML parsing with exact EDGAR acceptance timestamps
- SEC submissions metadata parsing for live Form 4 discovery
- transaction-code intent classification
- 10b5-1 filing flag preservation
- canonical company IDs anchored to SEC CIK
- immutable raw storage and DuckDB normalized event storage

### Milestone 3 — market data

- Tiingo EOD collection and normalization
- raw + adjusted OHLCV, dividends, and split factors
- conservative SEC issuer -> Tiingo resolution with ticker-reuse protection
- point-in-time security/ticker intervals
- dense DuckDB market storage
- point-in-time market features using only completed prior sessions
- conservative next-regular-session execution timing
- forward return, benchmark alpha, MFE, and MAE labels
- candidate snapshots that keep model inputs separate from future outcomes

### Milestone 4 — lobbying, contracts, and semantic extraction

- LDA `dt_posted` point-in-time lobbying events
- USAspending transaction-level contract events with explicit safe observation timestamps
- verified external source aliases that fail closed on canonical-company conflicts
- local Qwen3.5-4B semantic extraction with strict controlled topics, validation, versioning, and cache
- rolling contract/lobbying and cross-source convergence features

### Milestone 5 — LightGBM and portfolio backtesting

- point-in-time model-ready training rows
- 20-trading-day alpha, downside, and positive-alpha LightGBM models
- deterministic opportunity scoring
- strict annual walk-forward splits with matured-label boundaries
- long-only fixed-percentage portfolio simulation with costs and capacity limits
- score-percentile, feature-importance, and outlier-dependence reporting

## First historical experiment workflow

Install the project first:

```bash
python -m pip install -e ".[dev]"
```

### 1. Populate SEC insiders + verified Tiingo market history

Set the Tiingo token only in the environment:

```bash
export TIINGO_API_TOKEN="..."
```

Then populate the normalized stores:

```bash
python -m stock_trading.experiments.prepare \
  --data-root data \
  --start-year 2012 \
  --sec-user-agent "Stock-traiding your-contact@example.com"
```

This also stores SPY under the stable benchmark company ID `benchmark_spy`, writes an SEC company-identity manifest, and records unresolved/dead historical tickers instead of guessing them.

For a small smoke run before a full Tiingo backfill:

```bash
python -m stock_trading.experiments.prepare \
  --data-root data \
  --start-year 2024 \
  --max-unique-tickers 50 \
  --sec-user-agent "Stock-traiding your-contact@example.com"
```

### 2. Enrich historical LDA filings with local Qwen

Run a local OpenAI-compatible Qwen3.5-4B endpoint at `http://127.0.0.1:8000/v1` or set `QWEN_BASE_URL`.
An LDA API token is optional; when available it can be supplied through `LDA_API_TOKEN`.

```bash
python -m stock_trading.experiments.enrich \
  --data-root data \
  --start-year 2012
```

Only LDA clients that uniquely match a canonical SEC company name are automatically mapped and sent to Qwen. Ambiguous/subsidiary identities remain unresolved for explicit review.

### 3. Run the point-in-time LightGBM walk-forward experiment

```bash
python -m stock_trading.experiments.lightgbm \
  --events-db data/normalized/events.duckdb \
  --market-db data/normalized/market.duckdb \
  --benchmark-company-id benchmark_spy \
  --output-dir artifacts/experiments/first-run \
  --first-test-year 2018
```

The result bundle contains database hashes, frozen training rows, yearly model files, score buckets, portfolio results, feature importance, best-trade-removal stress tests, and aggregate walk-forward reporting.

## Point-in-time rule

Information is usable only after its documented public timestamp. Historical USAspending `action_date` is not assumed to be a historical publication timestamp, so historical contract records are not silently injected into backtests without a trustworthy visibility time. Live observed USAspending events remain supported.

## Development

```bash
pytest
```

GitHub Actions runs the complete dependency-backed test suite on pull requests.
