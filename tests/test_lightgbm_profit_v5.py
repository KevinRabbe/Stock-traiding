import numpy as np

from stock_trading.experiments.lightgbm_profit_v5 import _choose_horizons, _parser


def test_choose_horizons_uses_best_eligible_signal_then_expected_return() -> None:
    signals = {
        5: np.asarray([0.90, 0.80, 0.70]),
        20: np.asarray([0.95, 0.80, 0.99]),
        60: np.asarray([0.85, 0.80, 0.60]),
    }
    expected_returns = {
        5: np.asarray([0.02, 0.03, 0.01]),
        20: np.asarray([0.04, 0.02, 0.05]),
        60: np.asarray([0.03, 0.04, 0.06]),
    }
    eligible = {
        5: np.asarray([True, True, False]),
        20: np.asarray([True, True, False]),
        60: np.asarray([True, True, False]),
    }

    choice, choice_signal, any_eligible = _choose_horizons(
        signals,
        expected_returns,
        eligible,
    )

    assert choice.tolist() == [20, 60, -1]
    assert choice_signal.tolist() == [0.95, 0.80, 0.0]
    assert any_eligible.tolist() == [True, True, False]


def test_profit_v5_parser_defaults() -> None:
    args = _parser().parse_args(
        [
            "--experiment-dir",
            "data/experiment",
            "--market-db",
            "data/market.duckdb",
            "--benchmark-security-id",
            "benchmark_spy",
        ]
    )
    assert args.validation_top_fraction == 0.05
    assert args.alpha_rank_weight == 0.25
    assert args.calibration_window_days == 365
    assert args.max_expected_downside == 0.06
    assert args.allocation_pct == 0.02
    assert args.max_open_positions == 15
    assert args.round_trip_cost_bps == 20.0
    assert args.market_read_cache_series == 160
