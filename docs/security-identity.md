# Security identity and legal-company identity

Market history is keyed by a traded **security**, not by a legal company.

This distinction is required for point-in-time correctness. A continuous traded
security can survive a reincorporation, merger, holding-company change, or other
legal reorganization that causes the SEC CIK to change. Conversely, a ticker can
later be reused by an unrelated security. Treating `company_id`, `ticker`, and a
price series as the same identity silently corrupts historical attribution.

## Identity layers

The market pipeline therefore keeps three separate concepts:

1. **Company identity** — canonical legal/entity identity, currently anchored to
   SEC CIK (`company_id`). Sparse events such as insider transactions, lobbying,
   and government contracts attach here.
2. **Security identity** — one provider-observed traded-security history
   (`security_id`). Dense OHLCV, dividends, splits, labels, and market features
   attach here.
3. **Ticker** — a point-in-time market symbol carried by a security. It is a
   locator/alias, not durable identity.

The intended join is:

```text
sparse event
  -> company_id
  -> verified company/security mapping at decision time
  -> security_id
  -> security_market_daily
```

## Tiingo security IDs

For Tiingo EOD data, `security_id` is deterministic from:

```text
tiingo-eod | normalized ticker | Tiingo history start date
```

The legal company name and SEC CIK are deliberately excluded. This allows legal
successors that refer to the same continuous Tiingo history to share one
security without duplicating bars.

The history start date is deliberately included. If a ticker is later reused
for a different Tiingo history, it receives a different security ID rather than
being silently merged with the old instrument. Active-history end dates are not
included because they move forward over time.

## Invariants

- `security_market_daily` is keyed by `(security_id, date)`.
- Dense market bars contain no authoritative `company_id`.
- Multiple legal companies may map to the same `security_id`.
- One ticker may not map to two different security IDs over overlapping provider
  history intervals.
- A company may have more than one security mapping in storage. Candidate
  construction fails closed if more than one security is active for that company
  at the decision date until an explicit primary/security-class rule exists.
- Company-name mismatch is not promoted across different SEC CIKs merely because
  the ticker is the same. Corporate succession still requires evidence.
- Existing pre-migration `market_daily(company_id, ...)` data is legacy only. It
  can be copied into security-keyed storage *after* an explicit mapping resolves;
  it never becomes identity authority itself.

## AVGO example

The physical 20-ticker smoke exposed three SEC identities around `AVGO`:

- Avago Technologies LTD
- Broadcom Ltd
- Broadcom Inc.

The old design attempted to assign one Tiingo ticker history to exactly one SEC
company and therefore rejected the overlapping mappings. The new design permits
Broadcom legal-successor CIKs to reference the same security history while the
older Avago name mismatch remains unresolved. No cross-CIK succession is guessed.

This is the desired failure mode: preserve the market series once, preserve each
legal entity separately, and admit a company/security relationship only when the
available evidence supports it.
