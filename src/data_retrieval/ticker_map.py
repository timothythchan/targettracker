"""
ticker_map.py — Ticker resolution helpers (port of NB05/NB06 logic).

The CIQ ``ciq_transcripts.parquet`` is keyed on a numeric ``companyid`` and
sometimes carries a ``ticker`` column directly. When ``ticker`` is missing
or blank, the company name occasionally lets us recover the ticker via a
hand-curated map of S&P 500 mega-caps.

This module factors that resolution out of NB06 Cell 7 / Cell 10 so that
``scripts/build_demo_cache.py`` and any future scripts can reuse it
without copying the dictionary or the heuristics.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional


# Hand-curated companyname → ticker map covering the S&P 500 mega-caps that
# the demo focuses on. Derived from NB06 Cell 7. Not exhaustive — extend as
# needed when adding new tickers to ``DEMO_PAIRS``.
DEFAULT_NAME_TO_TICKER: Dict[str, str] = {
    "Apple Inc.":                                  "AAPL",
    "Microsoft Corporation":                       "MSFT",
    "Amazon.com, Inc.":                            "AMZN",
    "Alphabet Inc.":                               "GOOGL",
    "Meta Platforms, Inc.":                        "META",
    "Tesla, Inc.":                                 "TSLA",
    "NVIDIA Corporation":                          "NVDA",
    "Broadcom Inc.":                               "AVGO",
    "Oracle Corporation":                          "ORCL",
    "Cisco Systems, Inc.":                         "CSCO",
    "Accenture plc":                               "ACN",
    "Texas Instruments Incorporated":              "TXN",
    "QUALCOMM Incorporated":                       "QCOM",
    "Intel Corporation":                           "INTC",
    "Advanced Micro Devices, Inc.":                "AMD",
    "International Business Machines Corporation": "IBM",
    "Salesforce, Inc.":                            "CRM",
    "Adobe Inc.":                                  "ADBE",
    "JPMorgan Chase & Co.":                        "JPM",
    "Bank of America Corporation":                 "BAC",
    "The Goldman Sachs Group, Inc.":               "GS",
    "Morgan Stanley":                              "MS",
    "BlackRock, Inc.":                             "BLK",
    "The Charles Schwab Corporation":              "SCHW",
    "Visa Inc.":                                   "V",
    "Mastercard Incorporated":                     "MA",
    "AT&T Inc.":                                   "T",
}


def normalise_ticker(
    row: Mapping[str, object],
    *,
    candidate_tickers: Optional[Iterable[str]] = None,
    name_to_ticker: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Return a best-effort ticker for a CIQ row.

    Resolution order (matches NB06 Cell 7 ``_normalise_ticker``):

    1. ``row['ticker']`` when present and non-empty (uppercased, stripped).
    2. ``name_to_ticker[row['companyname']]`` exact match.
    3. Substring match: any ticker in ``candidate_tickers`` that appears
       inside the uppercased companyname.

    Returns an empty string when no ticker can be resolved. Callers should
    typically filter rows where the result is empty.

    Parameters
    ----------
    row:
        Mapping with optional ``ticker`` and ``companyname`` keys (typical
        ``pandas.Series`` from ``df.iterrows`` or ``df.apply(..., axis=1)``).
    candidate_tickers:
        Tickers to try as substring matches against ``companyname``. When
        omitted, only steps 1 and 2 run.
    name_to_ticker:
        Override the default ``{companyname: ticker}`` map. Defaults to
        ``DEFAULT_NAME_TO_TICKER``.
    """
    name_map = name_to_ticker if name_to_ticker is not None else DEFAULT_NAME_TO_TICKER

    raw_ticker = row.get("ticker")
    if isinstance(raw_ticker, str) and raw_ticker.strip():
        return raw_ticker.strip().upper()

    company_name = row.get("companyname")
    if isinstance(company_name, str) and company_name in name_map:
        return name_map[company_name]

    if candidate_tickers and isinstance(company_name, str) and company_name:
        company_upper = company_name.upper()
        for ticker in candidate_tickers:
            if ticker and ticker in company_upper:
                return ticker

    return ""


def build_id_to_ticker_map(
    ciq_id_map_df,
    *,
    candidate_tickers: Iterable[str],
    name_to_ticker: Optional[Mapping[str, str]] = None,
) -> Dict[float, str]:
    """
    Build a ``{companyid (float) -> ticker (str)}`` lookup from a CIQ
    ``(companyid, ticker, companyname)`` DataFrame.

    Mirrors NB06 Cell 10: each unique ``companyid`` is resolved to a ticker
    via :func:`normalise_ticker`, and only IDs whose ticker is in
    ``candidate_tickers`` are kept. The companyid is cast to ``float`` to
    match NB02's parquet output where ``companyid`` is stored as a float.

    Returns
    -------
    Dict mapping ``float(companyid)`` to uppercase ticker.
    """
    candidates = set(candidate_tickers)
    name_map = name_to_ticker if name_to_ticker is not None else DEFAULT_NAME_TO_TICKER

    id_map: Dict[float, str] = {}
    for _, row in ciq_id_map_df.iterrows():
        ticker = normalise_ticker(
            row,
            candidate_tickers=candidates,
            name_to_ticker=name_map,
        )
        if ticker and ticker in candidates:
            try:
                id_map[float(row["companyid"])] = ticker
            except (TypeError, ValueError):
                continue
    return id_map


def normalise_company_id(value: object) -> str:
    """
    Strip the trailing ``.0`` that pandas adds when company IDs were stored
    as floats. Mirrors NB06 ``_norm_cid``.

    >>> normalise_company_id(112350.0)
    '112350'
    >>> normalise_company_id('18711.0')
    '18711'
    >>> normalise_company_id(None)
    ''
    """
    if value is None:
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s
