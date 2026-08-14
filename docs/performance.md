# Historical pipeline performance

The historical experiment is intentionally point-in-time correct, but correctness does not require repeated database round-trips for the same immutable market series.

## Read-through market cache

`DuckDbMarketStore` maintains a small read-through LRU cache for historical bar series. The first PIT lookup for a security loads its ordered daily series from DuckDB; subsequent `next_bar_after`, `bar_on`, `bars_before`, and `bars_from` calls use binary search over the cached dates.

The cache is bounded (`read_cache_size`, default 8 securities), so a full-universe run does not need to keep every security in RAM. Historical dataset construction already processes opportunities in company/date order, so a small cache retains the current security plus the benchmark while avoiding nearly all repeated market queries for that company.

Writes and legacy migrations invalidate affected cached series. Company/security mapping lookups are also cached by company and invalidated when a mapping changes.

## Why this matters

Before this cache, building one opportunity could open DuckDB repeatedly for security resolution, next execution bar, benchmark alignment, feature history, and forward labels. With thousands of opportunities this created hundreds of thousands of tiny queries and repeated conversion of the same bars into Python objects.

The cache turns that pattern into roughly one ordered series load per security encountered (subject to LRU eviction), followed by in-memory `bisect` lookups.

This is an intermediate optimization. The next scale step is to precompute daily market features/forward labels and persist model-ready opportunity partitions in Parquet so most repeated experiments never need raw-bar reconstruction at all.
