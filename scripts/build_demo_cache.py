#!/usr/bin/env python
"""
build_demo_cache.py — Build the Gradio demo cache without opening a notebook
(port of NB06 ``06_demo_preparation_v2.ipynb``).

The Gradio app at ``demo/app.py`` reads three files from
``data/cache/demo/``:

    pipeline_cache.json     — full agent pipeline output per (ticker, quarter)
    portfolio_screen.json   — per-quarter ranked tables for Tab 2
    spacy_results.json      — sliced spaCy targets + MT (auxiliary)
    llm_results.json        — sliced LLM targets (auxiliary, fast-path)

Historically these were materialised by Notebook 08 in Colab. This script
reproduces every cell of NB06 (Cells 5, 7, 8, 10, 12, 13, 18) so the cache
can be built from the command line or via the app's Workflow tab.

Required inputs
---------------
    data/raw/ciq_transcripts.parquet           — from scripts/run_data_retrieval.py
    data/processed/spacy_targets.parquet       — from scripts/run_spacy_baseline.py
    data/processed/spacy_mt_scores.parquet     — from scripts/run_spacy_baseline.py
    data/processed/llm_targets.parquet         — from scripts/run_llm_extraction.py
                                                  (and optionally
                                                  scripts/run_rag_matching.py)

If ``llm_targets.parquet`` is missing but ``llm_targets.jsonl`` is present,
the script can repair the parquet on the fly via
``--repair-llm-parquet`` (NB06 Cell 5's bug fix).

Output
------
A populated ``data/cache/demo/`` directory consumable by the Gradio app.
The Gradio app's ``demo/sample_cache/`` synthetic stubs become unused once
this real cache exists.

Usage
-----
    python scripts/build_demo_cache.py
    python scripts/build_demo_cache.py --pairs AAPL=2023Q4 MSFT=2023Q4
    python scripts/build_demo_cache.py --repair-llm-parquet --skip-pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("scripts.build_demo_cache")


# Default demo universe — same 12 (ticker, quarter) pairs NB06 uses for the
# headline demo. The Gradio app deliberately scopes itself to a curated
# subset rather than the full S&P 200 fan-out.
DEFAULT_PAIRS: List[Tuple[str, str]] = [
    (t, q)
    for t in ["AAPL", "T", "MSFT", "NVDA", "META", "GOOGL"]
    for q in ["2020Q4", "2023Q4"]
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_demo_cache",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Project data root (default: ./data).",
    )
    p.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        metavar="TICKER=QUARTER",
        help=(
            "Optional override for the demo universe. Pass one or more "
            "TICKER=YYYYQN tokens. Defaults to the NB06 12-pair grid."
        ),
    )
    p.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help=(
            "Persistent ChromaDB directory used by the comparator. "
            "Defaults to <data-dir>/cache/chromadb_experiment."
        ),
    )
    p.add_argument(
        "--collection",
        default="targets_experiment",
        help="ChromaDB collection name (default: targets_experiment).",
    )
    p.add_argument(
        "--repair-llm-parquet",
        action="store_true",
        help=(
            "Rebuild llm_targets.parquet from llm_targets.jsonl using the "
            "NB06 Cell 5 fix that preserves trailing-zero company IDs."
        ),
    )
    p.add_argument(
        "--skip-pipeline",
        action="store_true",
        help=(
            "Build only the lightweight cache files (spacy_results.json, "
            "llm_results.json, portfolio_screen.json) and skip the "
            "LangGraph pipeline_cache.json rebuild. Useful for re-slicing "
            "after an upstream parquet refresh."
        ),
    )
    p.add_argument(
        "--reset-pipeline-cache",
        action="store_true",
        help=(
            "Wipe pipeline_cache.json before the run. Default is to keep "
            "existing entries (cache resume) and only fill missing pairs."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _parse_pairs(specs: Optional[List[str]]) -> List[Tuple[str, str]]:
    if not specs:
        return list(DEFAULT_PAIRS)
    pairs: List[Tuple[str, str]] = []
    for token in specs:
        if "=" not in token:
            raise SystemExit(f"--pairs value '{token}' must be TICKER=QUARTER")
        ticker, quarter = token.split("=", 1)
        pairs.append((ticker.strip().upper(), quarter.strip().upper()))
    return pairs


def _slice_spacy(
    data_processed: Path,
    cache_dir: Path,
    *,
    id_to_ticker: Dict[float, str],
    demo_quarters: List[str],
    demo_companies: List[str],
) -> Tuple[List[Dict[str, Any]], "pd.DataFrame"]:
    """NB06 Cell 10 — slice spaCy outputs to the demo universe + write JSON."""
    import pandas as pd

    spacy_targets_path = data_processed / "spacy_targets.parquet"
    spacy_mt_path      = data_processed / "spacy_mt_scores.parquet"

    if not spacy_targets_path.exists() or not spacy_mt_path.exists():
        raise FileNotFoundError(
            "spaCy NB02 outputs missing. Run scripts/run_spacy_baseline.py "
            f"to materialise {spacy_targets_path} and {spacy_mt_path}."
        )

    spacy_targets_full = pd.read_parquet(spacy_targets_path)
    spacy_mt_full      = pd.read_parquet(spacy_mt_path)
    logger.info(
        "Loaded spacy_targets (%d rows), spacy_mt_scores (%d rows)",
        len(spacy_targets_full), len(spacy_mt_full),
    )

    demo_companyids = set(id_to_ticker.keys())
    spacy_targets_full["companyid"] = pd.to_numeric(
        spacy_targets_full["companyid"], errors="coerce"
    )
    demo_tgt = spacy_targets_full[
        spacy_targets_full["companyid"].isin(demo_companyids)
    ].copy()
    demo_tgt["quarter"] = (
        demo_tgt["fiscalyear"].astype("Int64").astype(str)
        + "Q"
        + demo_tgt["fiscalquarter"].astype("Int64").astype(str)
    )
    demo_tgt = demo_tgt[demo_tgt["quarter"].isin(demo_quarters)].copy()
    demo_tgt["ticker"] = demo_tgt["companyid"].astype(float).map(id_to_ticker)

    # Slice MT scores
    spacy_mt_full["company_id"] = pd.to_numeric(
        spacy_mt_full["company_id"], errors="coerce"
    )
    demo_mt = spacy_mt_full[
        spacy_mt_full["company_id"].isin(demo_companyids)
    ].copy()
    if not demo_mt.empty and not demo_mt["quarter"].astype(str).str.match(r"^\d{4}Q[1-4]$").all():
        demo_mt["quarter"] = (
            demo_mt["fiscal_year"].astype("Int64").astype(str)
            + "Q"
            + demo_mt["fiscal_quarter"].astype("Int64").astype(str)
        )
    demo_mt = demo_mt[demo_mt["quarter"].isin(demo_quarters)].copy()
    demo_mt["ticker"] = demo_mt["company_id"].astype(float).map(id_to_ticker)

    # NB06 Cell 10 renames first, THEN selects columns. The previous order
    # ('keep mt_score, rename to mt_spacy, slice by keep_cols') triggered a
    # KeyError because ``mt_score`` no longer existed after the rename.
    demo_mt = demo_mt.rename(columns={"mt_score": "mt_spacy"})
    keep_cols = [c for c in [
        "ticker", "company_id", "quarter", "mt_spacy",
        "n_targets", "n_targets_tm4", "n_dropped", "n_new",
        "dropped_targets", "financial_drop_ratio", "nonfinancial_drop_ratio",
    ] if c in demo_mt.columns]
    spacy_mt_df = demo_mt[keep_cols]

    # Build per-(ticker, quarter) target list
    spacy_results: List[Dict[str, Any]] = []
    if not demo_tgt.empty:
        name_col = "normalized_text" if "normalized_text" in demo_tgt.columns else "target_text"
        for (tkr, qtr), grp in demo_tgt.groupby(["ticker", "quarter"], sort=False):
            seen = set()
            targets = []
            for _, r in grp.iterrows():
                nm = r.get(name_col, "")
                if not isinstance(nm, str) or not nm.strip():
                    continue
                canon = nm.strip().lower()
                if canon in seen:
                    continue
                seen.add(canon)
                targets.append({
                    "metric_name":    nm.strip(),
                    "target_text":    str(r.get("target_text", nm)),
                    "canonical_name": canon,
                })
            spacy_results.append({
                "ticker":  str(tkr).upper(),
                "quarter": str(qtr).upper(),
                "targets": targets,
            })

    cache_dir.mkdir(parents=True, exist_ok=True)
    spacy_cache_path = cache_dir / "spacy_results.json"
    # NB06 writes the rows with ``mt_spacy`` as the score column name (and
    # leaves it that way for any external consumer). The Gradio app only
    # reads portfolio_screen.json, so this file is informational, but we
    # follow the notebook's schema rather than re-renaming.
    with open(spacy_cache_path, "w") as fh:
        json.dump(
            {
                "results":   spacy_results,
                "mt_scores": spacy_mt_df.to_dict(orient="records"),
            },
            fh,
            default=str,
        )
    logger.info("Wrote %s (%d ticker-quarter entries)", spacy_cache_path, len(spacy_results))
    return spacy_results, spacy_mt_df


def _slice_llm(
    data_processed: Path,
    cache_dir: Path,
    *,
    id_to_ticker: Dict[float, str],
    demo_quarters: List[str],
) -> List[Dict[str, Any]]:
    """NB06 Cell 12 — slice llm_targets.parquet to the demo universe."""
    import pandas as pd

    parquet_path = data_processed / "llm_targets.parquet"
    if not parquet_path.exists():
        logger.warning(
            "llm_targets.parquet missing at %s — pipeline_cache will fall back "
            "to live LLM calls for any demo pair that needs extraction.",
            parquet_path,
        )
        return []

    llm_full = pd.read_parquet(parquet_path)
    garbage = [c for c in llm_full.columns
               if c.startswith("temporal_") and c != "temporal_framing"]
    if garbage:
        llm_full = llm_full.drop(columns=garbage)

    llm_full["company_id_f"] = pd.to_numeric(llm_full["company_id"], errors="coerce")
    demo_ids_f = set(id_to_ticker.keys())

    demo_llm = llm_full[
        llm_full["company_id_f"].isin(demo_ids_f)
        & llm_full["quarter"].isin(demo_quarters)
    ].copy()
    demo_llm["ticker"] = demo_llm["company_id_f"].map(id_to_ticker)

    target_fields = [
        "metric_name", "raw_text", "numerical_value", "trend_direction",
        "unit", "temporal_framing", "is_financial", "confidence",
        "canonical_name", "component_type", "numerical_direction",
        "trend_unit", "trend_direction_vs_prev_qtr", "unit_vs_prev_qtr",
    ]

    llm_results: List[Dict[str, Any]] = []
    for (tkr, qtr), grp in demo_llm.groupby(["ticker", "quarter"], sort=False):
        tid = str(grp["transcript_id"].iloc[0])
        targets: List[Dict[str, Any]] = []
        for _, r in grp.iterrows():
            t: Dict[str, Any] = {}
            for fld in target_fields:
                if fld not in grp.columns:
                    continue
                v = r[fld]
                if pd.isna(v):
                    t[fld] = None
                else:
                    t[fld] = v.item() if hasattr(v, "item") else v
            targets.append(t)
        llm_results.append({
            "ticker":        str(tkr).upper(),
            "quarter":       str(qtr).upper(),
            "transcript_id": tid,
            "targets":       targets,
        })

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "llm_results.json"
    payload = {
        "model":   os.environ.get("EARNINGSLENS_LLM_MODEL", "gemini-2.5-flash-lite"),
        "source":  "NB03 llm_targets.parquet (cached)",
        "n_rows":  int(len(demo_llm)),
        "results": llm_results,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, default=str)
    logger.info("Wrote %s (%d ticker-quarter entries)", out_path, len(llm_results))
    return llm_results


async def _run_pipeline_for_pairs(
    pairs: List[Tuple[str, str]],
    *,
    demo_transcripts: List[Dict[str, Any]],
    ticker_to_companyid: Dict[str, str],
    llm_results: List[Dict[str, Any]],
    pipeline_cache_path: Path,
    per_quarter_dir: Path,
    reset: bool,
) -> Dict[str, Dict[str, Any]]:
    """NB06 Cell 13 — run the LangGraph pipeline once per (ticker, quarter)."""
    from src.agents.graph import build_graph

    pipeline_cache: Dict[str, Dict[str, Any]] = {}
    if not reset and pipeline_cache_path.exists():
        try:
            pipeline_cache = json.loads(pipeline_cache_path.read_text())
            logger.info("Loaded existing pipeline_cache (%d entries)", len(pipeline_cache))
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse existing pipeline_cache (%s); starting fresh", exc)

    transcript_idx: Dict[Tuple[str, str], Dict[str, Any]] = {
        (tr["ticker"], tr["quarter"]): tr for tr in demo_transcripts
    }
    llm_idx: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
        (r["ticker"], r["quarter"]): r["targets"] for r in llm_results
    }

    app_graph = build_graph()
    per_quarter_dir.mkdir(parents=True, exist_ok=True)

    for ticker, quarter in pairs:
        cache_key = f"{ticker}_{quarter}"
        if cache_key in pipeline_cache and pipeline_cache[cache_key].get("historical_targets"):
            logger.info("[%s] cached — skipping", cache_key)
            continue

        transcript_obj = transcript_idx.get((ticker, quarter))
        if transcript_obj is None:
            logger.warning("[%s] no transcript loaded; skipping", cache_key)
            continue

        company_id = ticker_to_companyid.get(ticker)
        if company_id is None:
            logger.warning("[%s] no companyid mapping; skipping", cache_key)
            continue

        pre_extracted = llm_idx.get((ticker, quarter), [])
        initial_state = {
            "transcript":            {"components": transcript_obj["components"]},
            "company_id":            company_id,
            "ticker":                ticker,
            "fiscal_quarter":        quarter,
            "pre_extracted_targets": pre_extracted,
            "errors":                [],
        }

        t0 = time.time()
        try:
            result = await app_graph.ainvoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] graph.ainvoke failed: %s", cache_key, exc)
            continue

        payload = {
            "ticker":                  ticker,
            "quarter":                 quarter,
            "company_id":              company_id,
            "extracted_targets":       result.get("extracted_targets", []),
            "spacy_baseline_targets":  result.get("spacy_baseline_targets", []),
            "historical_targets":      result.get("historical_targets", []),
            "continuity_results":      result.get("continuity_results", {}),
            "classification_results":  result.get("classification_results", {}),
            "report":                  result.get("report", {}),
            "errors":                  result.get("errors", []),
        }
        pipeline_cache[cache_key] = payload
        with open(per_quarter_dir / f"{cache_key}.json", "w") as fh:
            json.dump(payload, fh, default=str)
        with open(pipeline_cache_path, "w") as fh:
            json.dump(pipeline_cache, fh, default=str)

        logger.info(
            "[%s] OK ext=%d hist=%d pre=%d (%.1fs)",
            cache_key,
            len(payload["extracted_targets"]),
            len(payload["historical_targets"]),
            len(pre_extracted),
            time.time() - t0,
        )

    return pipeline_cache


def _build_portfolio_screen(
    pipeline_cache: Dict[str, Dict[str, Any]],
    *,
    pairs: List[Tuple[str, str]],
    demo_quarters: List[str],
    demo_companies: List[str],
    spacy_mt_path: Path,
    ticker_to_companyid: Dict[str, str],
    cache_dir: Path,
) -> Path:
    """NB06 Cell 18 — build portfolio_screen.json from the pipeline cache."""
    import pandas as pd
    from src.data_retrieval.ticker_map import normalise_company_id

    # NB06 Cell 18 reloads the FULL spacy_mt_scores.parquet here (not the
    # demo-sliced df from _slice_spacy) and keys the lookup on the parquet's
    # own (company_id, quarter). Reproducing that exactly: when a ticker has
    # multiple CIQ companyids in the data (legacy entities), keying on
    # ticker_to_companyid[ticker] would attribute every row to the FIRST
    # mapped CIQ id and silently mis-key the MT score.
    spacy_lookup: Dict[Tuple[str, str], float] = {}
    if spacy_mt_path.exists():
        spacy_full = pd.read_parquet(spacy_mt_path)
        if not spacy_full.empty:
            for _, row in spacy_full.iterrows():
                cid_clean = normalise_company_id(row.get("company_id"))
                quarter = str(row.get("quarter") or "")
                if not cid_clean or not quarter:
                    continue
                spacy_lookup[(cid_clean, quarter)] = float(
                    row.get("mt_score", 0.0) or 0.0
                )

    pair_set = set(pairs)
    portfolio: Dict[str, List[Dict[str, Any]]] = {}

    for quarter in demo_quarters:
        rows: List[Dict[str, Any]] = []
        for ticker in demo_companies:
            if (ticker, quarter) not in pair_set:
                continue
            entry = pipeline_cache.get(f"{ticker}_{quarter}", {}) or {}
            cls = entry.get("classification_results", {}) or {}
            cont = entry.get("continuity_results", {}) or {}

            n_dropped = int(
                cls.get(
                    "n_dropped",
                    len(cls.get("dropped_financial", []) or [])
                    + len(cls.get("dropped_non_financial", []) or []),
                )
                or 0
            )
            risk_score = float(cls.get("risk_score", 0.0) or 0.0)
            risk_flag = cls.get("risk_flag", "unknown") or "unknown"
            company_id = normalise_company_id(
                entry.get("company_id", ticker_to_companyid.get(ticker, ""))
            )
            mt_score_spacy = spacy_lookup.get((company_id, quarter), 0.0)
            mt_score_llm_continuity = float(cont.get("mt_score", 0.0) or 0.0)

            rows.append({
                "ticker":                   ticker,
                "company_id":               company_id,
                "risk_score":               risk_score,
                "risk_flag":                risk_flag,
                "mt_score":                 mt_score_spacy,
                "mt_score_spacy":           mt_score_spacy,
                "mt_score_llm_continuity":  mt_score_llm_continuity,
                "n_dropped":                n_dropped,
                "n_extracted":              len(entry.get("extracted_targets", []) or []),
                "n_hist":                   len(entry.get("historical_targets", []) or []),
            })

        rows.sort(key=lambda r: r["risk_score"], reverse=True)
        portfolio[quarter] = rows

    out_path = cache_dir / "portfolio_screen.json"
    out_path.write_text(json.dumps(portfolio, indent=2))
    logger.info("Wrote %s (%d quarters)", out_path, len(portfolio))
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_dir = args.data_dir
    raw_dir = data_dir / "raw"
    data_processed = data_dir / "processed"
    cache_dir = data_dir / "cache" / "demo"
    chroma_dir = args.chroma_dir or (data_dir / "cache" / "chromadb_experiment")

    cache_dir.mkdir(parents=True, exist_ok=True)
    per_quarter_dir = cache_dir / "per_quarter"
    pipeline_cache_path = cache_dir / "pipeline_cache.json"

    # ChromaDB env vars consumed by src.agents.comparator_agent
    os.environ.setdefault("EARNINGSLENS_CHROMA_DIR", str(chroma_dir))
    os.environ.setdefault("EARNINGSLENS_CHROMA_COLLECTION", args.collection)

    pairs = _parse_pairs(args.pairs)
    demo_companies = sorted({t for t, _ in pairs})
    demo_quarters = sorted({q for _, q in pairs})
    logger.info("Demo universe: %d pairs (%d tickers x %d quarters)",
                len(pairs), len(demo_companies), len(demo_quarters))

    # Optional NB06 Cell 5 repair pass
    if args.repair_llm_parquet:
        from src.llm_extraction.extraction_pipeline import repair_parquet_from_jsonl
        jsonl_path = data_processed / "llm_targets.jsonl"
        parquet_path = data_processed / "llm_targets.parquet"
        stats = repair_parquet_from_jsonl(str(jsonl_path), str(parquet_path))
        logger.info("Repaired llm_targets.parquet: %s", stats)

    # NB06 Cells 7-8 — load + filter CIQ transcripts to the demo universe.
    from src.data_retrieval.ciq_loader import load_demo_subset, load_company_id_map
    ciq_path = raw_dir / "ciq_transcripts.parquet"
    demo_df, demo_transcripts = load_demo_subset(ciq_path, pairs)
    logger.info("Loaded %d demo transcripts.", len(demo_transcripts))

    id_to_ticker = load_company_id_map(ciq_path, candidate_tickers=demo_companies)
    ticker_to_companyid: Dict[str, str] = {}
    for cid_float, ticker in id_to_ticker.items():
        if ticker not in ticker_to_companyid:
            ticker_to_companyid[ticker] = str(int(cid_float))

    missing_map = sorted(set(demo_companies) - set(ticker_to_companyid))
    if missing_map:
        logger.warning("No CIQ companyid for tickers: %s", missing_map)

    # NB06 Cell 10
    spacy_results, spacy_mt_df = _slice_spacy(
        data_processed, cache_dir,
        id_to_ticker=id_to_ticker,
        demo_quarters=demo_quarters,
        demo_companies=demo_companies,
    )

    # NB06 Cell 12
    llm_results = _slice_llm(
        data_processed, cache_dir,
        id_to_ticker=id_to_ticker,
        demo_quarters=demo_quarters,
    )

    # NB06 Cell 13 — pipeline_cache.json
    if args.skip_pipeline:
        if pipeline_cache_path.exists():
            pipeline_cache = json.loads(pipeline_cache_path.read_text())
            logger.info("Reusing existing pipeline_cache (%d entries)", len(pipeline_cache))
        else:
            pipeline_cache = {}
            logger.warning("No pipeline_cache.json; portfolio_screen will be sparse.")
    else:
        if args.reset_pipeline_cache and pipeline_cache_path.exists():
            pipeline_cache_path.unlink()
            logger.info("Wiped %s", pipeline_cache_path)
        pipeline_cache = asyncio.run(
            _run_pipeline_for_pairs(
                pairs,
                demo_transcripts=demo_transcripts,
                ticker_to_companyid=ticker_to_companyid,
                llm_results=llm_results,
                pipeline_cache_path=pipeline_cache_path,
                per_quarter_dir=per_quarter_dir,
                reset=args.reset_pipeline_cache,
            )
        )

    # NB06 Cell 18 — portfolio_screen.json
    _build_portfolio_screen(
        pipeline_cache,
        pairs=pairs,
        demo_quarters=demo_quarters,
        demo_companies=demo_companies,
        spacy_mt_path=data_processed / "spacy_mt_scores.parquet",
        ticker_to_companyid=ticker_to_companyid,
        cache_dir=cache_dir,
    )

    # Final summary
    logger.info("=" * 70)
    logger.info("DEMO CACHE SUMMARY")
    for fname in (
        "spacy_results.json",
        "llm_results.json",
        "pipeline_cache.json",
        "portfolio_screen.json",
    ):
        fpath = cache_dir / fname
        if fpath.exists():
            logger.info("  %-25s %8.1f KB", fname, fpath.stat().st_size / 1024)
        else:
            logger.info("  %-25s MISSING", fname)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
