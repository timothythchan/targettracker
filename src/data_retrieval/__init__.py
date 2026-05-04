"""
data_retrieval
==============
EarningsLens WRDS data-retrieval module.

End-to-end mirror of NB01 v2 — every pull is filtered to the **top-200
S&P 500 universe** ranked by average market cap over 2010-2023.

Modules
-------
universe          — Build top-N S&P 500 universe (driver of the whole pipeline)
returns           — CRSP daily + monthly for the top-200 permnos
fundamentals      — Compustat quarterly for the top-200 gvkeys
analyst_forecasts — IBES quarterly summary statistics + SUE
factors           — Fama-French 5-factor + momentum monthly series
linkers           — Identifier links (CCM, S&P 500 historical, CIQ map)
transcripts       — Capital IQ earnings-call transcripts (batched fetch)
pipeline          — DataPipeline orchestrator

Quick start
-----------
>>> from data_retrieval.pipeline import DataPipeline
>>> with DataPipeline(wrds_username="your_username") as pipe:
...     pipe.run_all(output_dir="data/raw")
...     pipe.sanity_check("data/raw")

Or run individual modules from the command line, e.g.::

    python -m data_retrieval.universe       --output_dir data/raw
    python -m data_retrieval.returns        --output_dir data/raw
    python -m data_retrieval.fundamentals   --output_dir data/raw
    python -m data_retrieval.analyst_forecasts --output_dir data/raw
    python -m data_retrieval.factors        --output_dir data/raw
    python -m data_retrieval.transcripts    --output_dir data/raw --reuse_universe
"""

from .analyst_forecasts import (
    compute_sue,
    fetch_ibes_summary,
    save_analyst_forecasts,
)
from .factors import fetch_ff5_with_momentum, save_factors
from .fundamentals import fetch_quarterly_fundamentals, save_fundamentals
from .linkers import (
    attach_ticker_to_ciq_map,
    fetch_ccm_linktable,
    fetch_ciq_company_map,
    fetch_sp500_history,
    save_ciq_map,
)
from .pipeline import DataPipeline
from .returns import fetch_daily_returns, save_returns, subset_monthly_to_universe
from .transcripts import (
    assemble_transcripts,
    build_full_transcript_dataset,
    fetch_transcript_components,
    fetch_transcript_metadata,
    save_transcripts,
    save_transcript_meta,
)
from .universe import (
    END_DATE,
    N_FIRMS,
    RANKING_END,
    RANKING_START,
    RETURNS_END_DATE,
    START_DATE,
    build_top_n_universe,
    fetch_crsp_monthly_mcap,
    load_universe,
    save_universe,
)

__all__ = [
    # universe
    "build_top_n_universe", "fetch_crsp_monthly_mcap", "save_universe",
    "load_universe",
    "START_DATE", "END_DATE", "RETURNS_END_DATE",
    "RANKING_START", "RANKING_END", "N_FIRMS",
    # returns
    "fetch_daily_returns", "save_returns", "subset_monthly_to_universe",
    # fundamentals
    "fetch_quarterly_fundamentals", "save_fundamentals",
    # analyst forecasts
    "fetch_ibes_summary", "compute_sue", "save_analyst_forecasts",
    # factors
    "fetch_ff5_with_momentum", "save_factors",
    # linkers
    "fetch_ccm_linktable", "fetch_sp500_history",
    "fetch_ciq_company_map", "attach_ticker_to_ciq_map", "save_ciq_map",
    # transcripts
    "fetch_transcript_metadata", "save_transcript_meta",
    "fetch_transcript_components",
    "assemble_transcripts", "save_transcripts",
    "build_full_transcript_dataset",
    # pipeline
    "DataPipeline",
]
