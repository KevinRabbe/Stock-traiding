from stock_trading.experiments.lightgbm_profit_v4 import _parser


def test_profit_v4_parser_defaults() -> None:
    args = _parser().parse_args(["--experiment-dir", "data/experiment"])
    assert args.validation_top_fraction == 0.05
    assert args.alpha_rank_weight == 0.25
    assert args.calibration_window_days == 365
    assert args.max_expected_downside == 0.06
    assert args.allocation_pct == 0.02
    assert args.max_open_positions == 15
    assert args.round_trip_cost_bps == 20.0
