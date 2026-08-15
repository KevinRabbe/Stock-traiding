from importlib import import_module


_EXPORTS = {
    "BENCHMARK_SPY_COMPANY_ID": (".prepare", "BENCHMARK_SPY_COMPANY_ID"),
    "HistoricalExperimentConfig": (".lightgbm", "HistoricalExperimentConfig"),
    "HistoricalExperimentResult": (".lightgbm", "HistoricalExperimentResult"),
    "HistoricalUniverseResult": (".historical_universe", "HistoricalUniverseResult"),
    "LightGbmDiagnosticsResult": (".lightgbm_diagnostics", "LightGbmDiagnosticsResult"),
    "PermutationNullResult": (
        ".lightgbm_permutation_null",
        "PermutationNullResult",
    ),
    "ProfitRollingRankExperimentResult": (
        ".lightgbm_profit_rolling_rank",
        "ProfitRollingRankExperimentResult",
    ),
    "ProfitTargetExperimentResult": (
        ".lightgbm_profit",
        "ProfitTargetExperimentResult",
    ),
    "ProfitTrancheExperimentResult": (
        ".lightgbm_profit_tranches",
        "ProfitTrancheExperimentResult",
    ),
    "ValidationRankBacktestResult": (
        ".lightgbm_validation_rank",
        "ValidationRankBacktestResult",
    ),
    "LdaEnrichmentConfig": (".enrich", "LdaEnrichmentConfig"),
    "LdaEnrichmentResult": (".enrich", "LdaEnrichmentResult"),
    "SecMarketPopulationConfig": (".prepare", "SecMarketPopulationConfig"),
    "SecMarketPopulationResult": (".prepare", "SecMarketPopulationResult"),
    "build_historical_universe": (".historical_universe", "build_historical_universe"),
    "enrich_lda_and_qwen": (".enrich", "enrich_lda_and_qwen"),
    "load_historical_universe_company_ids": (
        ".historical_universe",
        "load_historical_universe_company_ids",
    ),
    "populate_sec_and_market": (".prepare", "populate_sec_and_market"),
    "run_historical_experiment": (".lightgbm", "run_historical_experiment"),
    "run_lightgbm_diagnostics": (".lightgbm_diagnostics", "run_lightgbm_diagnostics"),
    "run_permutation_null_test": (
        ".lightgbm_permutation_null",
        "run_permutation_null_test",
    ),
    "run_profit_rolling_rank_experiment": (
        ".lightgbm_profit_rolling_rank",
        "run_profit_rolling_rank_experiment",
    ),
    "run_profit_target_experiment": (
        ".lightgbm_profit",
        "run_profit_target_experiment",
    ),
    "run_profit_tranche_experiment": (
        ".lightgbm_profit_tranches",
        "run_profit_tranche_experiment",
    ),
    "run_validation_rank_backtest": (
        ".lightgbm_validation_rank",
        "run_validation_rank_backtest",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
