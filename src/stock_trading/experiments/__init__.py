from importlib import import_module


_EXPORTS = {
    "BENCHMARK_SPY_COMPANY_ID": (".prepare", "BENCHMARK_SPY_COMPANY_ID"),
    "HistoricalExperimentConfig": (".lightgbm", "HistoricalExperimentConfig"),
    "HistoricalExperimentResult": (".lightgbm", "HistoricalExperimentResult"),
    "LightGbmDiagnosticsResult": (".lightgbm_diagnostics", "LightGbmDiagnosticsResult"),
    "PermutationNullResult": (
        ".lightgbm_permutation_null",
        "PermutationNullResult",
    ),
    "ProfitTargetExperimentResult": (
        ".lightgbm_profit",
        "ProfitTargetExperimentResult",
    ),
    "ValidationRankBacktestResult": (
        ".lightgbm_validation_rank",
        "ValidationRankBacktestResult",
    ),
    "LdaEnrichmentConfig": (".enrich", "LdaEnrichmentConfig"),
    "LdaEnrichmentResult": (".enrich", "LdaEnrichmentResult"),
    "SecMarketPopulationConfig": (".prepare", "SecMarketPopulationConfig"),
    "SecMarketPopulationResult": (".prepare", "SecMarketPopulationResult"),
    "enrich_lda_and_qwen": (".enrich", "enrich_lda_and_qwen"),
    "populate_sec_and_market": (".prepare", "populate_sec_and_market"),
    "run_historical_experiment": (".lightgbm", "run_historical_experiment"),
    "run_lightgbm_diagnostics": (".lightgbm_diagnostics", "run_lightgbm_diagnostics"),
    "run_permutation_null_test": (
        ".lightgbm_permutation_null",
        "run_permutation_null_test",
    ),
    "run_profit_target_experiment": (
        ".lightgbm_profit",
        "run_profit_target_experiment",
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
