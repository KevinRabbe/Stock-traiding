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

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

GitHub Actions runs the complete dependency-backed test suite on pull requests.
