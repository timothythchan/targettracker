"""
ciq_loader.py — Load + slice CIQ transcripts to a focused (ticker, quarter) demo
subset (port of NB05 Cell 11 / NB06 Cell 7-8).

The full ``ciq_transcripts.parquet`` produced by ``run_data_retrieval`` is
~10k transcripts × hundreds of components. The Gradio app only needs the
~12 (ticker, quarter) pairs that the demo cache covers, so we filter
aggressively before any embedding/LLM call.

This module exposes one entry point, :func:`load_demo_subset`, which:

1. Reads the columns of interest from ``ciq_transcripts.parquet``.
2. Normalises the ``quarter`` column to ``YYYYQN`` (or derives it from
   ``event_date`` when absent).
3. Filters to component types ``{2, 3, 4}`` (presentation, analyst Q,
   management answer).
4. Drops rows with empty ``componenttext``.
5. Resolves a ticker for each row via :func:`ticker_map.normalise_ticker`.
6. Restricts to the exact ``(ticker, quarter)`` tuples in ``demo_pairs``.
7. Groups by ``(transcriptid, ticker, quarter)`` and emits a list of
   transcript dicts shaped for ``LLMTargetExtractor.extract_transcript`` /
   the LangGraph ``extractor_node``.

Each output transcript dict has:

    {
      "transcript_id": "AAPL_2023Q4",
      "ticker":        "AAPL",
      "quarter":       "2023Q4",
      "text":          "<concatenated component text>",
      "components": [
          {
              "text":              "...",
              "component_type":    2,
              "componenttext":     "...",   # raw alias preserved
              "component_type_id": 2,        # raw alias preserved
          },
          ...
      ],
    }

Both naming conventions (``text``/``component_type`` AND
``componenttext``/``component_type_id``) are populated so the same payload
flows cleanly through both the extractor (which reads ``text`` /
``component_type``) and any debug code that reads the raw CIQ keys.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq

from .ticker_map import DEFAULT_NAME_TO_TICKER, normalise_ticker

logger = logging.getLogger(__name__)


KEEP_CTYPES: Set[int] = {2, 3, 4}
"""CIQ component_type IDs we keep: 2=Presentation, 3=Analyst Q, 4=Mgmt Answer."""

_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")

_DESIRED_COLUMNS = (
    "companyid",
    "companyname",
    "ticker",
    "transcriptid",
    "year",
    "quarter",
    "component_type_id",
    "componenttext",
    "componentorder",
    "event_date",
)


def load_demo_subset(
    parquet_path: Path,
    demo_pairs: Sequence[Tuple[str, str]],
    *,
    name_to_ticker: Optional[Mapping[str, str]] = None,
    keep_component_types: Iterable[int] = KEEP_CTYPES,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Load a focused (ticker, quarter) subset from a CIQ transcripts parquet.

    Parameters
    ----------
    parquet_path:
        Path to ``data/raw/ciq_transcripts.parquet`` produced by
        :mod:`src.data_retrieval`.
    demo_pairs:
        Iterable of ``(ticker, quarter)`` tuples. Quarter strings must be
        in ``YYYYQN`` form (e.g. ``"2023Q4"``).
    name_to_ticker:
        Optional override for the companyname → ticker fallback map. Pass
        an extended dict when adding tickers beyond
        :data:`ticker_map.DEFAULT_NAME_TO_TICKER`.
    keep_component_types:
        CIQ component_type_id values to keep. Defaults to ``{2, 3, 4}``.

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict[str, Any]]]
        ``(demo_df, demo_transcripts)`` where ``demo_df`` is one row per
        transcript with metadata, and ``demo_transcripts`` is the list of
        component-level dicts described in this module's docstring.

    Raises
    ------
    FileNotFoundError
        When ``parquet_path`` does not exist.
    ValueError
        When the parquet is missing required columns or when quarter
        cannot be derived.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"CIQ parquet not found at {parquet_path}. "
            f"Run scripts/run_data_retrieval.py first."
        )

    name_map = (
        name_to_ticker if name_to_ticker is not None else DEFAULT_NAME_TO_TICKER
    )
    keep_ctypes = set(int(c) for c in keep_component_types)

    pair_set: Set[Tuple[str, str]] = {
        (str(t).upper(), str(q).upper()) for t, q in demo_pairs
    }
    if not pair_set:
        return pd.DataFrame(), []

    demo_companies: Set[str] = {t for t, _ in pair_set}
    demo_quarters:  Set[str] = {q for _, q in pair_set}

    pf = pq.ParquetFile(parquet_path)
    available_columns = set(pf.schema_arrow.names)

    required = {"transcriptid", "componenttext"}
    missing = required - available_columns
    if missing:
        raise ValueError(
            f"CIQ parquet at {parquet_path} is missing required columns: "
            f"{sorted(missing)}"
        )

    columns_to_load = [c for c in _DESIRED_COLUMNS if c in available_columns]
    raw_df = pd.read_parquet(parquet_path, columns=columns_to_load)
    logger.info("Loaded %d CIQ rows from %s", len(raw_df), parquet_path)

    raw_df = _attach_quarter(raw_df)
    raw_df = raw_df[raw_df["quarter"].isin(demo_quarters)].copy()

    if "component_type_id" in raw_df.columns:
        raw_df["component_type_id"] = pd.to_numeric(
            raw_df["component_type_id"], errors="coerce"
        ).astype("Int64")
        raw_df = raw_df[raw_df["component_type_id"].isin(keep_ctypes)].copy()
    else:
        raw_df["component_type_id"] = 4

    raw_df = raw_df[
        raw_df["componenttext"].notna()
        & raw_df["componenttext"].astype(str).str.strip().astype(bool)
    ].copy()

    raw_df["ticker_resolved"] = raw_df.apply(
        lambda row: normalise_ticker(
            row,
            candidate_tickers=demo_companies,
            name_to_ticker=name_map,
        ),
        axis=1,
    )
    raw_df = raw_df[raw_df["ticker_resolved"].isin(demo_companies)].copy()

    raw_df["_pair"] = list(zip(raw_df["ticker_resolved"], raw_df["quarter"]))
    raw_df = raw_df[raw_df["_pair"].isin(pair_set)].copy()
    raw_df = raw_df.drop(columns=["_pair"])

    sort_cols = [
        c for c in ("transcriptid", "componentorder") if c in raw_df.columns
    ]
    if sort_cols:
        raw_df = raw_df.sort_values(sort_cols)

    demo_transcripts: List[Dict[str, Any]] = []
    demo_rows: List[Dict[str, Any]] = []
    grouped = raw_df.groupby(
        ["transcriptid", "ticker_resolved", "quarter"],
        as_index=False,
        sort=False,
    )

    for (tid, ticker, quarter), grp in grouped:
        components: List[Dict[str, Any]] = []
        for _, row in grp.iterrows():
            text = str(row["componenttext"]).strip()
            if not text:
                continue
            ctype = int(row.get("component_type_id") or 0)
            components.append({
                "text":              text,
                "component_type":    ctype,
                # Preserve raw CIQ keys for any downstream code that reads them.
                "componenttext":     text,
                "component_type_id": ctype,
            })
        if not components:
            continue

        agg_text = "\n\n".join(c["text"] for c in components)
        transcript_id_str = f"{ticker}_{quarter}"

        demo_transcripts.append({
            "transcript_id": transcript_id_str,
            "ticker":        ticker,
            "quarter":       quarter,
            "text":          agg_text,
            "components":    components,
        })
        demo_rows.append({
            "transcript_id": transcript_id_str,
            "transcriptid":  int(tid) if pd.notna(tid) else None,
            "ticker":        ticker,
            "quarter":       quarter,
            "n_components":  len(components),
        })

    demo_df = pd.DataFrame(demo_rows)
    return demo_df, demo_transcripts


def load_company_id_map(
    parquet_path: Path,
    *,
    candidate_tickers: Iterable[str],
    name_to_ticker: Optional[Mapping[str, str]] = None,
) -> Dict[float, str]:
    """
    Build a ``{companyid (float) -> ticker}`` lookup from a CIQ parquet.

    Mirrors NB06 Cell 10's ``id_to_ticker`` build step. Float keys match
    NB02's ``spacy_targets.parquet`` (where ``companyid`` is float).
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"CIQ parquet not found at {parquet_path}. "
            f"Run scripts/run_data_retrieval.py first."
        )

    ciq_id_map = (
        pd.read_parquet(parquet_path, columns=["companyid", "ticker", "companyname"])
        .drop_duplicates(subset=["companyid"])
    )

    from .ticker_map import build_id_to_ticker_map  # local to avoid cycle

    return build_id_to_ticker_map(
        ciq_id_map,
        candidate_tickers=candidate_tickers,
        name_to_ticker=name_to_ticker,
    )


def _attach_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ``df`` has a ``quarter`` column in ``YYYYQN`` form. Derives it
    from ``event_date`` (calendar quarter) when no string ``quarter``
    column is present.
    """
    if "quarter" in df.columns:
        df["quarter"] = df["quarter"].astype(str).str.upper().str.strip()
        valid = df["quarter"].str.match(_QUARTER_RE.pattern)
        return df[valid].copy()

    if "event_date" in df.columns:
        df = df.copy()
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df = df.dropna(subset=["event_date"])
        df["quarter"] = (
            df["event_date"].dt.year.astype(str)
            + "Q"
            + df["event_date"].dt.quarter.astype(str)
        )
        return df

    raise ValueError(
        "CIQ parquet has no 'quarter' or 'event_date' column to derive "
        "fiscal-quarter from."
    )
