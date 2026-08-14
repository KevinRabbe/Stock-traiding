# Opportunity-level training rows

The canonical event store remains immutable and source-granular. SEC Form 4 transaction lines, lobbying filings, government-contract transactions, and future event families stay as separate canonical facts.

The ML dataset does **not** use those source rows as independent trading samples.

## Training unit

A model row represents one point-in-time trading opportunity:

```text
(company_id, next executable market session)
```

All model trigger events that are public before that same conservative execution session are aggregated into the opportunity. This prevents one filing with several transaction lines from becoming several nearly identical samples with the same forward label.

The daily/EOD execution policy is unchanged: information is acted on only at the next actual regular-session open after its Eastern publication date. Events from different calendar days can therefore merge when they still lead to the same next executable session, such as Friday/weekend disclosures before Monday open.

## Point-in-time construction

For each opportunity:

1. group raw model triggers by company and publication market date;
2. resolve one market snapshot per company/day;
3. merge day slices that resolve to the same company + execution date;
4. set the opportunity decision time to the latest trigger publication time in that group;
5. build market and historical event features using only information public by that decision time;
6. aggregate the newly public trigger facts for the opportunity;
7. attach one aligned 20-trading-day forward label.

The row ID is deterministic:

```text
opportunity:<company_id>:<execution_date>
```

`trigger_event_ids` keeps provenance back to every canonical source event that formed the row.

## Trigger aggregation

The opportunity trigger feature block includes:

- whether insider, contract, or lobbying triggers are present;
- event counts by family;
- unique actor count;
- summed and maximum source value;
- semantic count and maximum semantic scores;
- union of controlled semantic topics.

Existing rolling insider/contract/lobbying/cross-source features are still built from the full company history visible at the opportunity decision time.

## Company-balanced training

LightGBM uses company-balanced sample weights by default. Within each train or validation split, every legal company contributes equal total training mass regardless of how many opportunities it generated.

For company `c` with `n_c` rows in a split of `N` rows across `C` companies, each row receives weight:

```text
N / (C * n_c)
```

The mean row weight remains 1.0. This is intended to stop prolific filers from dominating model fitting while retaining every valid opportunity.

## What does not change

- canonical source events are never collapsed or deleted;
- company/security identity rules remain unchanged;
- point-in-time market mapping remains fail-closed;
- labels remain strictly forward and aligned to stock/benchmark trading sessions;
- Congress remains disabled unless explicitly enabled;
- portfolio/risk rules are unchanged.
