from stock_trading.experiments.lightgbm_profit_v7 import _parser


def test_v7_cli_defaults() -> None:
    args = _parser().parse_args(
        [
            "--experiment-dir",
            "data/experiments/example",
            "--market-db",
            "data/normalized/market.duckdb",
            "--benchmark-security-id",
            "benchmark_spy",
        ]
    )

    assert args.validation_top_fraction == 0.05
    assert args.alpha_rank_weight == 0.25
    assert args.base_allocation_pct == 0.02
    assert args.min_allocation_pct == 0.01
    assert args.max_gross_exposure_pct == 0.20
    assert args.correlation_lookback_sessions == 60
    assert args.high_correlation_threshold == 0.75
    assert args.max_correlated_exposure_pct == 0.08
