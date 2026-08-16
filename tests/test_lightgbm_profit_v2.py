from stock_trading.experiments.lightgbm_profit_v2 import _parser


def test_profit_v2_cli_imports_and_preserves_locked_defaults() -> None:
    args = _parser().parse_args(["--experiment-dir", "data/experiment"])

    assert args.validation_top_fraction == 0.05
    assert args.max_expected_downside == 0.06
    assert args.starting_capital == 10_000.0
    assert args.allocation_pct == 0.02
    assert args.max_open_positions == 15
    assert args.round_trip_cost_bps == 20.0
    assert args.permutations == 250
    assert args.seed == 42
