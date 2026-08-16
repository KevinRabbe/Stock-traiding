from stock_trading.experiments.strategy_engine_v5_replay import _parser


def test_strategy_engine_v5_replay_cli_defaults() -> None:
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
    assert args.calibration_window_days == 365
    assert args.max_expected_downside == 0.06
    assert args.starting_capital == 10_000.0
    assert args.allocation_pct == 0.02
    assert args.max_open_positions == 15
    assert args.round_trip_cost_bps == 20.0
    assert args.market_read_cache_series == 160
