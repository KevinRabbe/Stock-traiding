# Stock-traiding

Private trading-system research and implementation repository.

The system is being built around point-in-time-safe alternative data, semantic extraction, machine-learning opportunity ranking, deterministic risk controls, and deliberately simple execution.

## Core architecture

```text
Raw public data
    -> immutable RawRecord
    -> source-specific Normalizer
    -> canonical Event
    -> point-in-time feature state
    -> LightGBM + temporal trading model
    -> deterministic trade filter
    -> fixed-percentage risk/execution layer
```

Qwen is planned as a semantic extraction layer for text-heavy sources. Authoritative source fields and model-generated semantic annotations are kept separate.

## Milestone 1: data contracts

The current foundation defines:

- immutable raw records with SHA-256 verification
- deterministic event IDs for idempotent ingestion
- UTC-only internal timestamps; naive datetimes are rejected
- separate `event_time`, `public_time`, and `first_tradable_time`
- typed payloads for insider trades, contracts, lobbying, FX, Congress, market bars, and corporate actions
- semantic annotations that cannot overwrite source facts
- collector and normalizer protocols
- regression tests for leakage-critical invariants

The core rule is that an event cannot be used as information before `public_time`, and execution cannot occur before `first_tradable_time`.

## Development

Python 3.11+ is required.

```bash
python -m venv .venv
# activate the virtual environment
pip install -e ".[dev]"
pytest
```

Large datasets, model artifacts, caches, logs, and local secrets are intentionally excluded from Git.
