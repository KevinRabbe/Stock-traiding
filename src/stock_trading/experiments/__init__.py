from importlib import import_module


_EXPORTS = {
    "BENCHMARK_SPY_COMPANY_ID": (".prepare", "BENCHMARK_SPY_COMPANY_ID"),
    "HistoricalExperimentConfig": (".lightgbm", "HistoricalExperimentConfig"),
    "HistoricalExperimentResult": (".lightgbm", "HistoricalExperimentResult"),
    "LdaEnrichmentConfig": (".enrich", "LdaEnrichmentConfig"),
    "LdaEnrichmentResult": (".enrich", "LdaEnrichmentResult"),
    "SecMarketPopulationConfig": (".prepare", "SecMarketPopulationConfig"),
    "SecMarketPopulationResult": (".prepare", "SecMarketPopulationResult"),
    "enrich_lda_and_qwen": (".enrich", "enrich_lda_and_qwen"),
    "populate_sec_and_market": (".prepare", "populate_sec_and_market"),
    "run_historical_experiment": (".lightgbm", "run_historical_experiment"),
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
