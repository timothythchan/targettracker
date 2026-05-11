"""
transcripts.py
==============
Retrieve earnings-call transcripts from WRDS Capital IQ Transcripts schema
for the **top-200 universe**.

Mirrors §9 of NB01 v2.

Tables used:
    ciq_transcripts.wrds_transcript_detail   — transcript-level metadata
    ciq_transcripts.ciqtranscriptcomponent   — paragraph-level text components
    ciq.wrds_gvkey                            — gvkey ↔ CIQ companyid
                                                (via :mod:`linkers`)

Filters:
    keydeveventtypeid = 48                       (earnings calls)
    transcriptcomponenttypeid IN (2, 3, 4)       (Pres, Q, A)
    mostimportantdateutc BETWEEN START_DATE AND END_DATE
    companyid IN (top-200 CIQ companyids)

Steps:
    9a. Map gvkey → companyid via ciq.wrds_gvkey (in :mod:`linkers`).
    9b. Pull transcript metadata (filter early, dedup by created_date).
    9c. Pull components in batches of 500 transcripts (memory-safe).
    9d. Join meta + components, derive quarter/year columns, save the
        canonical ``ciq_transcripts.parquet``.

Outputs:
    data/raw/ciq_company_map.parquet     (saved by linkers.save_ciq_map)
    data/raw/ciq_transcript_meta.parquet
    data/raw/ciq_transcripts.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import wrds

from .linkers import (
    attach_ticker_to_ciq_map,
    fetch_ciq_company_map,
    save_ciq_map,
)
from .universe import END_DATE, START_DATE, build_top_n_universe, fetch_crsp_monthly_mcap

logger = logging.getLogger(__name__)

# Component-type code → human label (NB01 v2)
COMPONENT_LABELS = {2: "Mgmt Presentation", 3: "Analyst Question", 4: "Mgmt Answer"}
COMPONENT_TYPE_IDS = (2, 3, 4)
EARNINGS_CALL_KEYDEV_ID = 48
COMPONENT_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# 9b — transcript metadata
# ---------------------------------------------------------------------------


def fetch_transcript_metadata(
    db: wrds.Connection,
    companyids: Iterable[int],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    keydev_id: int = EARNINGS_CALL_KEYDEV_ID,
) -> pd.DataFrame:
    """Pull earnings-call transcript metadata, dedup to one per (company, event).

    Returns columns:
        companyid, companyname, transcriptid, keydeveventtypeid, keydevid,
        event_date, created_date
    """
    companyid_list = ",".join(str(int(c)) for c in companyids)
    if not companyid_list:
        raise ValueError("companyids is empty.")

    query = f"""
        SELECT d.companyid, d.companyname,
               d.transcriptid, d.keydeveventtypeid, d.keydevid,
               d.mostimportantdateutc        AS event_date,
               d.transcriptcreationdate_utc  AS created_date
        FROM ciq_transcripts.wrds_transcript_detail AS d
        WHERE d.companyid IN ({companyid_list})
          AND d.keydeveventtypeid = {keydev_id}
          AND d.mostimportantdateutc BETWEEN '{start_date}' AND '{end_date}'
    """
    n_companies = len(companyid_list.split(","))
    logger.info("Pulling transcript metadata for %d CIQ companies (%s → %s)...",
                n_companies, start_date, end_date)
    meta = db.raw_sql(query, date_cols=["event_date", "created_date"])

    # Keep the latest transcript per (companyid, event_date)
    meta = (
        meta.sort_values("created_date")
            .drop_duplicates(["companyid", "event_date"], keep="last")
            .reset_index(drop=True)
    )
    logger.info("Earnings-call transcripts: %d (avg %.1f / firm)",
                len(meta), len(meta) / max(meta["companyid"].nunique(), 1))
    return meta


def save_transcript_meta(meta: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ciq_transcript_meta.parquet"
    meta.to_parquet(out, index=False)
    logger.info("Saved transcript meta → %s", out)
    return out


# ---------------------------------------------------------------------------
# 9c — components in batches
# ---------------------------------------------------------------------------


def fetch_transcript_components(
    db: wrds.Connection,
    transcript_ids: Iterable[int],
    component_type_ids: Iterable[int] = COMPONENT_TYPE_IDS,
    batch_size: int = COMPONENT_BATCH_SIZE,
    progress: bool = True,
) -> pd.DataFrame:
    """Pull transcript components in batches of ``batch_size`` transcripts.

    componenttext can be very long; batching keeps memory bounded and
    prevents the SQL IN-list from exceeding driver limits.
    """
    transcript_ids = [int(x) for x in transcript_ids]
    if not transcript_ids:
        raise ValueError("transcript_ids is empty.")

    type_list = ",".join(str(t) for t in component_type_ids)

    iterator = range(0, len(transcript_ids), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Pulling components")
        except ImportError:
            pass

    chunks: List[pd.DataFrame] = []
    for i in iterator:
        batch_ids = transcript_ids[i:i + batch_size]
        id_list = ",".join(str(x) for x in batch_ids)
        query = f"""
            SELECT c.transcriptid, c.transcriptcomponentid, c.componentorder,
                   c.transcriptcomponenttypeid AS component_type_id,
                   c.componenttext
            FROM ciq_transcripts.ciqtranscriptcomponent AS c
            WHERE c.transcriptid IN ({id_list})
              AND c.transcriptcomponenttypeid IN ({type_list})
        """
        chunks.append(db.raw_sql(query))

    components = pd.concat(chunks, ignore_index=True)
    logger.info("Components: %d total (avg %.1f / transcript)",
                len(components), len(components) / max(len(transcript_ids), 1))
    return components


# ---------------------------------------------------------------------------
# 9d — assemble the canonical transcripts.parquet
# ---------------------------------------------------------------------------


def assemble_transcripts(
    components: pd.DataFrame,
    meta: pd.DataFrame,
    ciq_map: pd.DataFrame,
) -> pd.DataFrame:
    """Join components + meta + CIQ ticker map into the final table.

    Output columns (NB01 v2 ordering):
        transcriptid, companyid, companyname, ticker,
        event_date, year, quarter,
        component_type_id, component_label,
        componentorder, componenttext,
        keydeveventtypeid, keydevid, created_date,
        transcriptcomponentid
    """
    # Make sure the meta has ticker available
    if "ticker" not in meta.columns:
        meta = meta.merge(ciq_map[["companyid", "ticker"]], on="companyid", how="left")

    transcripts = components.merge(meta, on="transcriptid", how="left")
    transcripts["component_label"] = transcripts["component_type_id"].map(COMPONENT_LABELS)
    transcripts["quarter"] = (
        pd.to_datetime(transcripts["event_date"]).dt.to_period("Q").astype(str)
    )
    transcripts["year"] = pd.to_datetime(transcripts["event_date"]).dt.year

    final_cols = [
        "transcriptid", "companyid", "companyname", "ticker",
        "event_date", "year", "quarter",
        "component_type_id", "component_label",
        "componentorder", "componenttext",
        "keydeveventtypeid", "keydevid", "created_date",
        "transcriptcomponentid",
    ]
    transcripts = transcripts[final_cols]
    transcripts = (
        transcripts.sort_values(["companyid", "event_date", "componentorder"])
                   .reset_index(drop=True)
    )
    return transcripts


def save_transcripts(transcripts: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ciq_transcripts.parquet"
    transcripts.to_parquet(out, index=False)
    size_mb = out.stat().st_size / 1e6
    logger.info("Saved transcripts → %s (%.1f MB · %d components · %d transcripts · %d tickers)",
                out, size_mb, len(transcripts),
                transcripts["transcriptid"].nunique(),
                transcripts["ticker"].nunique())
    return out


# ---------------------------------------------------------------------------
# Convenience: full transcript pipeline for a given universe
# ---------------------------------------------------------------------------


def build_full_transcript_dataset(
    db: wrds.Connection,
    universe: pd.DataFrame,
    output_dir: str | Path,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    batch_size: int = COMPONENT_BATCH_SIZE,
) -> pd.DataFrame:
    """End-to-end §9 of NB01 v2: map → meta → components → assemble → save.

    Returns the assembled transcripts DataFrame (also persisted).
    """
    output_dir = Path(output_dir)

    # 9a — gvkey → companyid
    ciq_map = fetch_ciq_company_map(db, universe["gvkey"].tolist())
    ciq_map = attach_ticker_to_ciq_map(ciq_map, universe)
    save_ciq_map(ciq_map, output_dir)

    # 9b — metadata
    meta = fetch_transcript_metadata(
        db, ciq_map["companyid"].dropna().unique(),
        start_date=start_date, end_date=end_date,
    )
    meta = meta.merge(ciq_map[["companyid", "ticker"]], on="companyid", how="left")
    save_transcript_meta(meta, output_dir)

    # 9c — components (batched)
    components = fetch_transcript_components(
        db, meta["transcriptid"].astype(int).tolist(),
        batch_size=batch_size,
    )

    # 9d — assemble + save
    transcripts = assemble_transcripts(components, meta, ciq_map)
    save_transcripts(transcripts, output_dir)
    return transcripts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pull CIQ transcripts for top-200 universe.")
    parser.add_argument("--wrds_user", required=False)
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--start_date", default=START_DATE)
    parser.add_argument("--end_date", default=END_DATE)
    parser.add_argument("--batch_size", type=int, default=COMPONENT_BATCH_SIZE)
    parser.add_argument("--reuse_universe", action="store_true",
                        help="Load top200_universe.parquet from --output_dir "
                             "instead of rebuilding it.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")

    db = wrds.Connection(wrds_username=args.wrds_user) if args.wrds_user else wrds.Connection()
    try:
        if args.reuse_universe:
            universe = pd.read_parquet(Path(args.output_dir) / "top200_universe.parquet")
        else:
            crsp_m = fetch_crsp_monthly_mcap(db)
            universe = build_top_n_universe(db, crsp_monthly=crsp_m)

        build_full_transcript_dataset(
            db, universe, args.output_dir,
            start_date=args.start_date, end_date=args.end_date,
            batch_size=args.batch_size,
        )
    finally:
        db.close()


if __name__ == "__main__":
    _main()
