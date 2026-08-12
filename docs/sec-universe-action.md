# Manual SEC universe scan in GitHub Actions

The repository includes `.github/workflows/sec-universe.yml` so the SEC-only universe scan can run on a GitHub-hosted runner without a Tiingo token or local Qwen service.

## One-time setup

Create a repository Actions secret named `SEC_USER_AGENT` containing the identifying application/contact string used for SEC automated access. Keep the contact value in the secret rather than committing it to the public repository.

## Run the scan

1. Open the repository's **Actions** tab.
2. Select **SEC universe scan**.
3. Choose **Run workflow**.
4. Leave `start_year=2012` and `start_quarter=1` for the planned first full universe scan, or choose a later period for a smaller test.
5. Start the workflow.

The workflow uses `--sec-only`; it does not read `TIINGO_API_TOKEN` and does not call Tiingo.

## Results

The Actions job summary prints `sec_universe.json`, including:

- number of SEC companies observed,
- selected unique ticker count,
- total unique ticker count before any limit,
- estimated minimum Tiingo metadata/price requests for the later market backfill.

The run also uploads a `sec-universe-<year>q<quarter>` artifact containing the manifests for 14 days.

Use this result to decide whether to run the 50-symbol market smoke test or proceed directly to the full market history backfill.
