"""
data_retrieval
==============
EarningsLens WRDS data-retrieval module.

The public objects are loaded lazily so importing this package does not open or
require WRDS-related dependencies until a retrieval function is actually used.
"""

_EXPORTS = {
    # universe
    "build_top_n_universe": "universe",
    "fetch_crsp_monthly_mcap": "universe",
    "save_universe": "universe",
    "load_universe": "universe",
    "START_DATE": "universe",
    "END_DATE": "universe",
    "RETURNS_END_DATE": "universe",
    "RANKING_START": "universe",
    "RANKING_END": "universe",
    "N_FIRMS": "universe",
    # returns
    "fetch_daily_returns": "returns",
    "save_returns": "returns",
    "subset_monthly_to_universe": "returns",
    # fundamentals
    "fetch_quarterly_fundamentals": "fundamentals",
    "save_fundamentals": "fundamentals",
    # analyst forecasts
    "fetch_ibes_summary": "analyst_forecasts",
    "compute_sue": "analyst_forecasts",
    "save_analyst_forecasts": "analyst_forecasts",
    # factors
    "fetch_ff5_with_momentum": "factors",
    "save_factors": "factors",
    # linkers
    "fetch_ccm_linktable": "linkers",
    "fetch_sp500_history": "linkers",
    "fetch_ciq_company_map": "linkers",
    "attach_ticker_to_ciq_map": "linkers",
    "save_ciq_map": "linkers",
    # transcripts
    "fetch_transcript_metadata": "transcripts",
    "save_transcript_meta": "transcripts",
    "fetch_transcript_components": "transcripts",
    "assemble_transcripts": "transcripts",
    "save_transcripts": "transcripts",
    "build_full_transcript_dataset": "transcripts",
    # pipeline
    "DataPipeline": "pipeline",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)
