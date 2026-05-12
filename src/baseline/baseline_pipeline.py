"""
BaselinePipeline — Layer 1 End-to-End Orchestration
====================================================
Orchestrates the full spaCy baseline pipeline for the EarningsLens project:

  1. Load raw transcript data from  data/raw/ciq_transcripts.parquet
  2. Group transcripts by (companyid, fiscalyear, fiscalquarter)
  3. Run SpacyTargetExtractor on every transcript component
  4. Normalise extracted targets for cross-quarter matching
  5. Pass target sets to MovingTargetsComputer
  6. Optionally augment with persistence flags
  7. Save results:
       data/processed/spacy_targets.parquet   — all extracted targets
       data/processed/spacy_mt_scores.parquet — MT measures per (company, quarter)

Expected schema for  data/raw/ciq_transcripts.parquet
------------------------------------------------------
The loader accepts either normalized notebook columns:
  companyid, fiscalyear, fiscalquarter, component_type, text

or the WRDS CIQ retrieval output:
  companyid, year, quarter, component_type_id, componenttext

Optional columns (passed through to targets parquet):
  transcriptid, keydeveventid, transcriptpersonid, componentorder, etc.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_pipeline_components() -> Tuple[Any, Any, Any]:
    """Import heavy baseline components only when the pipeline actually runs."""
    try:
        from .moving_targets import MovingTargetsComputer, add_persistence_flags
        from .target_extractor import SpacyTargetExtractor
    except ImportError:
        # Allow direct execution: python baseline_pipeline.py
        import sys as _sys
        from pathlib import Path as _Path
        _src_root = str(_Path(__file__).resolve().parent.parent)
        _sys.path.insert(0, _src_root)
        from baseline.moving_targets import MovingTargetsComputer, add_persistence_flags  # type: ignore
        from baseline.target_extractor import SpacyTargetExtractor  # type: ignore

    return MovingTargetsComputer, add_persistence_flags, SpacyTargetExtractor

logger = logging.getLogger("earningslens.baseline.pipeline")

# ---------------------------------------------------------------------------
# Default paths  (relative to project root, i.e. earningslens/)
# ---------------------------------------------------------------------------

DEFAULT_ROOT = Path(__file__).resolve().parents[2]  # repository root
DEFAULT_RAW = DEFAULT_ROOT / "data" / "raw" / "transcripts.parquet"
DEFAULT_TARGETS_OUT = DEFAULT_ROOT / "data" / "processed" / "spacy_targets.parquet"
DEFAULT_MT_OUT = DEFAULT_ROOT / "data" / "processed" / "spacy_mt_scores.parquet"


# ---------------------------------------------------------------------------
# BaselinePipeline
# ---------------------------------------------------------------------------


class BaselinePipeline:
    """
    End-to-end Layer 1 pipeline: transcript → spaCy targets → MT scores.

    Parameters
    ----------
    spacy_model : str
        spaCy model name (default ``en_core_web_lg``).
    compute_persistence : bool
        Whether to compute the 12-quarter persistence flag (default True).
        This adds one extra column to the MT scores DataFrame but requires
        a backward-pass over target_sets which is moderately expensive.
    persistence_window : int
        Number of prior consecutive quarters to test for persistence (default 12).
    raw_path : Path | str
        Path to ``transcripts.parquet``.
    targets_out : Path | str
        Output path for per-target rows.
    mt_out : Path | str
        Output path for MT score rows.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        compute_persistence: bool = True,
        persistence_window: int = 12,
        raw_path: Optional[Path] = None,
        targets_out: Optional[Path] = None,
        mt_out: Optional[Path] = None,
    ) -> None:
        self.spacy_model = spacy_model
        self.compute_persistence = compute_persistence
        self.persistence_window = persistence_window

        self.raw_path = Path(raw_path) if raw_path else DEFAULT_RAW
        self.targets_out = Path(targets_out) if targets_out else DEFAULT_TARGETS_OUT
        self.mt_out = Path(mt_out) if mt_out else DEFAULT_MT_OUT

        # Lazy-initialised in run()
        self._extractor: Optional[Any] = None
        self._computer: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full pipeline.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            ``(targets_df, mt_df)`` — the two output DataFrames also persisted
            to parquet.
        """
        t0 = time.perf_counter()
        logger.info("=" * 70)
        logger.info("EarningsLens  Layer 1 Baseline Pipeline — START")
        logger.info("=" * 70)

        # Step 1: Load raw transcripts
        raw_df = self._load_transcripts()

        # Step 2: Initialise NLP components
        MovingTargetsComputer, add_persistence_flags_fn, SpacyTargetExtractor = (
            _load_pipeline_components()
        )
        self._extractor = SpacyTargetExtractor(model_name=self.spacy_model)
        self._computer = MovingTargetsComputer(
            persistence_window=self.persistence_window
        )

        # Step 3: Extract targets
        targets_df, target_sets = self._extract_all_targets(raw_df)

        # Step 4: Compute MT measures
        mt_df = self._compute_mt_measures(targets_df, target_sets)

        # Step 5: Optionally add persistence flags
        if self.compute_persistence and not mt_df.empty:
            logger.info("Computing persistence flags (window=%d quarters) …", self.persistence_window)
            mt_df = add_persistence_flags_fn(mt_df, target_sets, self.persistence_window)

        # Step 6: Save outputs
        self._save_outputs(targets_df, mt_df)

        elapsed = time.perf_counter() - t0
        logger.info("=" * 70)
        logger.info("Pipeline complete in %.1f s", elapsed)
        logger.info(
            "Outputs:\n  targets → %s\n  MT scores → %s",
            self.targets_out,
            self.mt_out,
        )
        logger.info("=" * 70)

        return targets_df, mt_df

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _load_transcripts(self) -> pd.DataFrame:
        """Load, normalize, and validate a raw transcripts parquet file.

        The WRDS retrieval pipeline writes ``ciq_transcripts.parquet`` with
        Capital IQ names such as ``componenttext``, ``component_type_id``,
        ``year``, and a string ``quarter`` (for example ``2023Q4``). Older
        notebook exports may already use the normalized baseline names. This
        loader accepts both schemas so the script pipeline can run directly on
        the retrieval output without a manual notebook conversion step.
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Raw transcript file not found: {self.raw_path}\n"
                "Please place the data file at the expected path or pass "
                "--raw-path/--input to the CLI."
            )

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas required: pip install pandas pyarrow") from exc

        logger.info("Loading transcripts from %s …", self.raw_path)
        df = pd.read_parquet(self.raw_path)
        logger.info("  Loaded %d rows, %d columns.", len(df), len(df.columns))

        df = self._normalize_transcript_schema(df)

        # Validate required columns
        required = {"companyid", "fiscalyear", "fiscalquarter", "component_type", "text"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"transcript input is missing required columns after normalization: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Coerce types
        df["fiscalyear"] = df["fiscalyear"].astype(int)
        df["fiscalquarter"] = df["fiscalquarter"].astype(int)
        df["component_type"] = df["component_type"].astype(int)
        df["companyid"] = df["companyid"].astype(str)
        df["text"] = df["text"].fillna("").astype(str)

        n_calls = df.groupby(["companyid", "fiscalyear", "fiscalquarter"]).ngroups
        logger.info(
            "  %d unique (company, quarter) pairs across %d companies.",
            n_calls,
            df["companyid"].nunique(),
        )
        return df

    @staticmethod
    def _normalize_transcript_schema(df: "pd.DataFrame") -> "pd.DataFrame":
        """Return *df* with baseline-compatible transcript column names."""
        import pandas as pd

        df = df.copy()
        rename_map = {
            "componenttext": "text",
            "component_type_id": "component_type",
            "year": "fiscalyear",
        }
        for source, target in rename_map.items():
            if target not in df.columns and source in df.columns:
                df = df.rename(columns={source: target})

        if "fiscalquarter" not in df.columns:
            if "quarter" in df.columns:
                quarter = df["quarter"].astype(str).str.extract(r"Q([1-4])", expand=False)
                df["fiscalquarter"] = quarter
                if "fiscalyear" not in df.columns:
                    year = df["quarter"].astype(str).str.extract(r"(\d{4})", expand=False)
                    df["fiscalyear"] = year
            elif "event_date" in df.columns:
                event_dates = pd.to_datetime(df["event_date"], errors="coerce")
                df["fiscalquarter"] = event_dates.dt.quarter
                if "fiscalyear" not in df.columns:
                    df["fiscalyear"] = event_dates.dt.year

        return df

    def _extract_all_targets(
        self, raw_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], List[Dict]]]:
        """
        Run SpacyTargetExtractor over all transcripts grouped by
        (companyid, fiscalyear, fiscalquarter).

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            ``(targets_df, target_sets)`` where ``target_sets`` is keyed
            by ``(company_id, quarter_key)``.
        """
        groups = raw_df.groupby(["companyid", "fiscalyear", "fiscalquarter"])
        n_groups = len(groups)
        logger.info("Extracting targets from %d (company, quarter) groups …", n_groups)

        all_target_rows: List[Dict] = []
        target_sets: Dict[Tuple[str, str], List[Dict]] = {}

        n_processed = 0
        n_errors = 0
        log_interval = max(1, n_groups // 20)  # log every ~5 %

        for (company_id, fy, fq), group_df in groups:
            try:
                components = [
                    {
                        "text": row["text"],
                        "component_type": row["component_type"],
                    }
                    for _, row in group_df.iterrows()
                ]

                raw_targets = self._extractor.extract_from_transcript(components)

                # Normalise each target and attach metadata
                enriched: List[Dict] = []
                for t in raw_targets:
                    norm = self._extractor.normalize_target(t["target_text"])
                    enriched.append(
                        {
                            **t,
                            "normalized_text": norm,
                            "companyid": str(company_id),
                            "fiscalyear": int(fy),
                            "fiscalquarter": int(fq),
                            "quarter": f"{fy}Q{fq}",
                        }
                    )
                    all_target_rows.append(enriched[-1])

                quarter_key = f"{fy}Q{fq}"
                target_sets[(str(company_id), quarter_key)] = enriched

                n_processed += 1
                if n_processed % log_interval == 0 or n_processed == n_groups:
                    logger.info(
                        "  Progress: %d/%d groups processed "
                        "(%d total targets so far, %d errors)",
                        n_processed,
                        n_groups,
                        len(all_target_rows),
                        n_errors,
                    )

            except Exception as exc:  # noqa: BLE001
                n_errors += 1
                logger.warning(
                    "Error processing group (%s, %sQ%s): %s",
                    company_id,
                    fy,
                    fq,
                    exc,
                )

        logger.info(
            "Extraction complete: %d groups processed, %d targets found, %d errors.",
            n_processed,
            len(all_target_rows),
            n_errors,
        )

        import pandas as pd

        targets_df = pd.DataFrame(all_target_rows) if all_target_rows else pd.DataFrame()
        return targets_df, target_sets

    def _compute_mt_measures(
        self,
        targets_df: pd.DataFrame,
        target_sets: Dict[Tuple[str, str], List[Dict]],
    ) -> pd.DataFrame:
        """Run MovingTargetsComputer and return the MT scores DataFrame."""
        if target_sets:
            logger.info(
                "Computing MT measures for %d (company, quarter) sets …",
                len(target_sets),
            )
            mt_df = self._computer.compute_mt(target_sets)
        else:
            import pandas as pd

            logger.warning("No target sets available — MT scores will be empty.")
            mt_df = pd.DataFrame()

        if not mt_df.empty:
            logger.info(
                "MT scores computed: %d rows, mean MT=%.4f, "
                "median MT=%.4f, fraction with MT>0=%.2f%%",
                len(mt_df),
                mt_df["mt_score"].mean(),
                mt_df["mt_score"].median(),
                100 * (mt_df["mt_score"] > 0).mean(),
            )
        return mt_df

    def _save_outputs(
        self, targets_df: pd.DataFrame, mt_df: pd.DataFrame
    ) -> None:
        """Persist targets and MT scores to parquet."""
        self.targets_out.parent.mkdir(parents=True, exist_ok=True)
        self.mt_out.parent.mkdir(parents=True, exist_ok=True)

        if not targets_df.empty:
            targets_df.to_parquet(self.targets_out, index=False)
            logger.info(
                "Saved %d target rows → %s", len(targets_df), self.targets_out
            )
        else:
            logger.warning("targets_df is empty — no file written to %s", self.targets_out)

        if not mt_df.empty:
            # Serialise list/dict columns as strings for broad parquet compatibility
            mt_df_save = mt_df.copy()
            for col in ("dropped_targets", "persistent_flag"):
                if col in mt_df_save.columns:
                    mt_df_save[col] = mt_df_save[col].astype(str)
            mt_df_save.to_parquet(self.mt_out, index=False)
            logger.info(
                "Saved %d MT score rows → %s", len(mt_df_save), self.mt_out
            )
        else:
            logger.warning("mt_df is empty — no file written to %s", self.mt_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "EarningsLens Layer 1 — spaCy Baseline Pipeline\n"
            "Replicates the Cohen & Nguyen (2024) Moving Targets methodology."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-path",
        "--input",
        dest="raw_path",
        type=Path,
        default=DEFAULT_RAW,
        help=f"Path to CIQ or normalized transcripts parquet (default: {DEFAULT_RAW})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for default output files. If supplied and --targets-out/"
            "--mt-out are not supplied, writes spacy_targets.parquet and "
            "spacy_mt_scores.parquet inside this directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for default output files. If supplied and --targets-out/"
            "--mt-out are not supplied, writes spacy_targets.parquet and "
            "spacy_mt_scores.parquet inside this directory."
        ),
    )
    parser.add_argument(
        "--targets-out",
        type=Path,
        default=DEFAULT_TARGETS_OUT,
        help=f"Output path for spacy_targets.parquet (default: {DEFAULT_TARGETS_OUT})",
    )
    parser.add_argument(
        "--mt-out",
        type=Path,
        default=DEFAULT_MT_OUT,
        help=f"Output path for spacy_mt_scores.parquet (default: {DEFAULT_MT_OUT})",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
        help="spaCy model name (default: en_core_web_lg)",
    )
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Skip the 12-quarter persistence flag computation.",
    )
    parser.add_argument(
        "--persistence-window",
        type=int,
        default=12,
        help="Number of prior quarters to test persistence over (default: 12).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for local, notebook-free baseline execution."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)-36s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    targets_out = args.targets_out
    mt_out = args.mt_out
    if args.output_dir is not None:
        if targets_out == DEFAULT_TARGETS_OUT:
            targets_out = args.output_dir / "spacy_targets.parquet"
        if mt_out == DEFAULT_MT_OUT:
            mt_out = args.output_dir / "spacy_mt_scores.parquet"

    pipeline = BaselinePipeline(
        spacy_model=args.spacy_model,
        compute_persistence=not args.no_persistence,
        persistence_window=args.persistence_window,
        raw_path=args.raw_path,
        targets_out=targets_out,
        mt_out=mt_out,
    )

    try:
        pipeline.run()
        return 0
    except FileNotFoundError as exc:
        logger.error("Input file error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in pipeline: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
