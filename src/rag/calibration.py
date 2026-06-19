"""
calibration.py — RAG threshold calibration via human-labeled pairs.

Direct port of NB04b (``04b_threshold_calibration.ipynb``) into a
reusable module. The notebook contains an annotated calibration of the
two cosine thresholds the production RAG matcher uses:

- ``maintained_threshold`` — separates *maintained* from *rephrased*.
- ``rephrased_threshold``  — separates *rephrased* from *dropped*
  (i.e. the lower bound of "tracked").

The calibration runs on a single labeled CSV
(``mt_calibration_sample_labeled.csv``) and emits two JSON artifacts:

- ``mt_calibration_result.json`` — full F1 sweep + logistic + bootstrap CI
- ``semantic_mt_scores_calibrated.meta.json`` — minimal sidecar pointing to
  the operational thresholds, consumed by ``scripts/run_rag_matching.py``.

This module exposes three pure functions and one orchestrator
(:func:`run_calibration`) so the script is a thin wrapper. All
plotting / matplotlib code from the notebook is intentionally omitted —
the operational artifact is the JSON, and analytical figures should be
regenerated from the bootstrap arrays returned in ``calibration_result``.

The methodology mirrors the notebook (single-rater protocol, weighted
F1 sweep, logistic-regression p=0.5 cross-over, non-parametric bootstrap
CI). See the NB04b markdown for the academic justification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default threshold values from the EarningsLens v1 release. Pre-calibration
# they were locked at these placeholders; calibrate_against_labels()
# computes the recommended replacements.
PLACEHOLDER_MAINTAINED = 0.65
PLACEHOLDER_REPHRASED  = 0.40

# Confidence weight scheme used in NB04b. Reviewers can override per call.
DEFAULT_CONFIDENCE_WEIGHTS: Dict[str, float] = {"H": 1.0, "M": 0.7, "L": 0.3}

VALID_LABELS = {"MAINTAINED", "REPHRASED", "DROPPED", "UNCLEAR"}
VALID_CONFIDENCE = {"H", "M", "L"}

EXPECTED_COLUMNS = (
    "pair_id", "company_id", "quarter",
    "current_metric", "current_text", "current_numval", "current_unit",
    "historical_quarter", "historical_metric", "historical_text",
    "historical_numval", "historical_unit",
    "similarity", "label", "confidence", "notes",
)


@dataclass
class CalibrationOutcome:
    """Per-boundary calibration result."""

    name: str
    placeholder: float
    calibrated_f1: float          # operational F1-optimal threshold
    calibrated_ba: float          # balanced-accuracy-optimal threshold
    p50_threshold: float          # logistic p=0.5 cross-over (-alpha/beta)
    logistic_beta: float
    logistic_alpha: float
    f1_at_placeholder: float
    f1_at_calibrated: float
    auc: float
    n: int
    grid: np.ndarray              # the 0..1 sweep grid
    f1_curve: np.ndarray
    ba_curve: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict, matching the NB04b output schema."""
        return {
            "name": self.name,
            "placeholder": float(self.placeholder),
            "calibrated_f1": float(self.calibrated_f1),
            "calibrated_ba": float(self.calibrated_ba),
            "p50_threshold": float(self.p50_threshold),
            "logistic": {
                "beta": float(self.logistic_beta),
                "intercept": float(self.logistic_alpha),
            },
            "f1_placeholder": float(self.f1_at_placeholder),
            "f1_calibrated": float(self.f1_at_calibrated),
            "auc": float(self.auc),
            "n": int(self.n),
        }


def load_labels(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate ``mt_calibration_sample_labeled.csv`` (NB04b Cell 5).

    - Asserts every expected column is present.
    - Upper-cases label and confidence.
    - Drops rows with ``label == 'UNCLEAR'``.
    - Adds a ``weight`` column from :data:`DEFAULT_CONFIDENCE_WEIGHTS`.

    Raises
    ------
    FileNotFoundError
        When the CSV does not exist.
    ValueError
        When required columns or value sets are violated.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Labeled calibration CSV not found at {csv_path}. "
            "It is produced by the NB04 sampling step (best-match-per-current "
            "with stratified cosine bins) and labeled by hand."
        )

    df = pd.read_csv(csv_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Calibration CSV is missing expected columns: {missing}"
        )

    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df["confidence"] = df["confidence"].astype(str).str.strip().str.upper()

    bad_label = sorted(set(df.loc[~df["label"].isin(VALID_LABELS), "label"]))
    bad_conf = sorted(set(df.loc[~df["confidence"].isin(VALID_CONFIDENCE), "confidence"]))
    if bad_label:
        raise ValueError(f"Unexpected label values: {bad_label}")
    if bad_conf:
        raise ValueError(f"Unexpected confidence values: {bad_conf}")

    n_unclear = int((df["label"] == "UNCLEAR").sum())
    if n_unclear:
        df = df[df["label"] != "UNCLEAR"].reset_index(drop=True)
    df["weight"] = df["confidence"].map(DEFAULT_CONFIDENCE_WEIGHTS).astype(float)

    logger.info(
        "Loaded %d labeled pairs (dropped %d UNCLEAR). Distribution: %s",
        len(df),
        n_unclear,
        df["label"].value_counts().to_dict(),
    )
    return df


def calibrate_one_boundary(
    similarity: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    placeholder: float,
    name: str,
    n_grid_points: int = 101,
) -> CalibrationOutcome:
    """
    Run the NB04b ``calibrate()`` routine for a single binary boundary.

    Parameters
    ----------
    similarity:
        1-D cosine similarities for each pair.
    y:
        Binary 0/1 labels.
    weights:
        Per-pair confidence weights.
    placeholder:
        v1 production threshold (used for the Δ-F1 reporting line).
    name:
        Human-readable boundary name (e.g. ``"maintained_threshold"``).

    Notes
    -----
    Imports ``sklearn`` lazily so the module is importable even when
    sklearn is not installed (the rest of the package does not need it).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            balanced_accuracy_score,
            f1_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for threshold calibration. "
            "Install with: pip install scikit-learn"
        ) from exc

    x = np.asarray(similarity, dtype=float).ravel()
    y = np.asarray(y, dtype=int)
    w = np.asarray(weights, dtype=float)
    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError("similarity, y, weights must be the same shape")

    grid = np.linspace(0.0, 1.0, n_grid_points)
    f1s = np.array([
        f1_score(y, (x >= t).astype(int), sample_weight=w, zero_division=0)
        for t in grid
    ])
    bas = np.array([
        balanced_accuracy_score(y, (x >= t).astype(int), sample_weight=w)
        for t in grid
    ])

    best_f1_idx = int(np.argmax(f1s))
    best_ba_idx = int(np.argmax(bas))

    lr = LogisticRegression(C=1e6, solver="liblinear")
    lr.fit(x.reshape(-1, 1), y, sample_weight=w)
    beta = float(lr.coef_[0, 0])
    alpha = float(lr.intercept_[0])
    p50 = float(-alpha / beta) if beta != 0 else float("nan")

    auc = float(roc_auc_score(y, x, sample_weight=w))

    f1_placeholder = float(
        f1_score(y, (x >= placeholder).astype(int), sample_weight=w, zero_division=0)
    )
    f1_calibrated = float(f1s[best_f1_idx])

    return CalibrationOutcome(
        name=name,
        placeholder=float(placeholder),
        calibrated_f1=float(grid[best_f1_idx]),
        calibrated_ba=float(grid[best_ba_idx]),
        p50_threshold=p50,
        logistic_beta=beta,
        logistic_alpha=alpha,
        f1_at_placeholder=f1_placeholder,
        f1_at_calibrated=f1_calibrated,
        auc=auc,
        n=int(len(y)),
        grid=grid,
        f1_curve=f1s,
        ba_curve=bas,
    )


def bootstrap_threshold_ci(
    similarity: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    n_iter: int = 1000,
    seed: int = 20260509,
    n_grid_points: int = 101,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Non-parametric bootstrap CI on the F1-optimal threshold.

    Returns ``(lower, upper, all_iter_thresholds)``. The third element is
    the full B-length array so the caller can report alternative quantiles
    or histograms without re-running the bootstrap.

    Pass an explicit ``rng`` (``np.random.Generator``) when running both
    boundaries off one shared stream — that matches NB04b Cell 17, which
    seeds one global ``np.random.default_rng(20260509)`` and consumes its
    output across both Maintained-vs-Dropped and Tracked-vs-Dropped runs.
    Without ``rng``, each call restarts from ``seed`` and the tracked CI
    will diverge from the notebook because the resampling sequence differs.
    """
    try:
        from sklearn.metrics import f1_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for threshold calibration."
        ) from exc

    if rng is None:
        rng = np.random.default_rng(seed)
    x = np.asarray(similarity, dtype=float).ravel()
    y = np.asarray(y, dtype=int)
    w = np.asarray(weights, dtype=float)
    grid = np.linspace(0.0, 1.0, n_grid_points)
    n = len(y)

    out = np.empty(n_iter, dtype=float)
    for b in range(n_iter):
        sample = rng.choice(n, size=n, replace=True)
        xs, ys, ws = x[sample], y[sample], w[sample]
        f1s = np.array([
            f1_score(ys, (xs >= t).astype(int), sample_weight=ws, zero_division=0)
            for t in grid
        ])
        out[b] = grid[int(np.argmax(f1s))]

    lo = float(np.percentile(out, ci_low))
    hi = float(np.percentile(out, ci_high))
    return lo, hi, out


def run_calibration(
    labels_df: pd.DataFrame,
    *,
    confidence_weights: Mapping[str, float] = DEFAULT_CONFIDENCE_WEIGHTS,
    placeholder_maintained: float = PLACEHOLDER_MAINTAINED,
    placeholder_rephrased: float = PLACEHOLDER_REPHRASED,
    bootstrap_iter: int = 1000,
    bootstrap_seed: int = 20260509,
) -> Dict[str, Any]:
    """
    Orchestrate the full NB04b pipeline and return a JSON-ready dict.

    Mirrors NB04b Cells 7-21 end-to-end:

    1. Apply ``confidence_weights``.
    2. Build Maintained-vs-Dropped (Problem A) and Tracked-vs-Dropped
       (Problem B) binary problems.
    3. Run :func:`calibrate_one_boundary` on each.
    4. Compute bootstrap CIs.
    5. Assemble the output dict in the schema NB04b writes to
       ``mt_calibration_result.json``.

    Parameters
    ----------
    labels_df:
        DataFrame as returned by :func:`load_labels` (must include
        ``similarity``, ``label``, ``confidence``).

    Returns
    -------
    Dict[str, Any]
        See :func:`save_calibration_result` for the schema.
    """
    df = labels_df.copy()
    df["weight"] = df["confidence"].map(dict(confidence_weights)).astype(float)
    if df["weight"].isna().any():
        raise ValueError(
            "Some 'confidence' values failed to map to a weight. "
            f"Got: {sorted(set(df['confidence']))} vs scheme {dict(confidence_weights)}"
        )

    # Problem A: Maintained vs Dropped — drop REPHRASED rows.
    mvd = df[df["label"].isin(["MAINTAINED", "DROPPED"])].copy()
    mvd_y = (mvd["label"] == "MAINTAINED").astype(int).values
    mvd_x = mvd["similarity"].values.astype(float)
    mvd_w = mvd["weight"].values.astype(float)

    # Problem B: Tracked (Maintained ∪ Rephrased) vs Dropped.
    tvd_y = df["label"].isin(["MAINTAINED", "REPHRASED"]).astype(int).values
    tvd_x = df["similarity"].values.astype(float)
    tvd_w = df["weight"].values.astype(float)

    res_maintained = calibrate_one_boundary(
        mvd_x, mvd_y, mvd_w,
        placeholder=placeholder_maintained,
        name="maintained_threshold",
    )
    res_tracked = calibrate_one_boundary(
        tvd_x, tvd_y, tvd_w,
        placeholder=placeholder_rephrased,
        name="tracked_threshold",
    )

    # Share one RNG stream across both boundaries — NB04b Cell 17 declares
    # ``RNG = np.random.default_rng(20260509)`` once and calls the bootstrap
    # back-to-back, so the tracked-boundary CI depends on the maintained
    # bootstrap's draws. Reseeding per-call would silently diverge.
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    mvd_lo, mvd_hi, _ = bootstrap_threshold_ci(
        mvd_x, mvd_y, mvd_w, n_iter=bootstrap_iter, rng=bootstrap_rng,
    )
    tvd_lo, tvd_hi, _ = bootstrap_threshold_ci(
        tvd_x, tvd_y, tvd_w, n_iter=bootstrap_iter, rng=bootstrap_rng,
    )

    return {
        "maintained_threshold": res_maintained.to_dict(),
        "tracked_threshold":    res_tracked.to_dict(),
        "sample": {
            "n_total":    int(len(df)),
            "maintained": int((df["label"] == "MAINTAINED").sum()),
            "rephrased":  int((df["label"] == "REPHRASED").sum()),
            "dropped":    int((df["label"] == "DROPPED").sum()),
        },
        "confidence_weights": dict(confidence_weights),
        "bootstrap_ci_95": {
            "maintained": [float(mvd_lo), float(mvd_hi)],
            "tracked":    [float(tvd_lo), float(tvd_hi)],
        },
        "placeholder_thresholds": {
            "maintained": float(placeholder_maintained),
            "rephrased":  float(placeholder_rephrased),
        },
        "notes": (
            "Single-rater calibration on labeled pairs sampled from "
            "per_pair_sims.parquet (stratified on cosine, t-4 lag only). "
            "Confidence weights applied via sklearn sample_weight on F1 sweep "
            "and LogisticRegression. F1-optimal threshold is the operational "
            "value; logistic p=0.5 reported as smoothness sanity check."
        ),
    }


def save_calibration_result(
    result: Dict[str, Any],
    *,
    result_path: Path,
    meta_path: Optional[Path] = None,
    labeled_csv: Optional[Path] = None,
) -> None:
    """
    Persist the calibration result + minimal meta sidecar.

    The ``result`` JSON is the canonical artifact consumed by
    ``scripts/run_rag_matching.py`` (and historically NB04). The optional
    ``meta_path`` writes a thin sidecar collocated with the (future)
    calibrated parquet, mirroring NB04b's
    ``semantic_mt_scores_calibrated.meta.json`` output.
    """
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Wrote calibration result: %s", result_path)

    if meta_path is None:
        return

    # Match NB04b Cell 21: write paths RELATIVE to the meta file's directory
    # (typically ``data/processed/``) so the artifact is portable across
    # machines and matches the notebook's reviewer-readable string.
    def _maybe_relative(target: Path, base: Path) -> str:
        try:
            return str(Path(target).relative_to(base))
        except (TypeError, ValueError):
            return str(target)

    meta_base = Path(meta_path).parent
    meta = {
        "thresholds": {
            "maintained": result["maintained_threshold"]["calibrated_f1"],
            "rephrased":  result["tracked_threshold"]["calibrated_f1"],
        },
        "calibration_source": _maybe_relative(result_path, meta_base),
        "labeled_csv": _maybe_relative(labeled_csv, meta_base) if labeled_csv else None,
        "lag": "t-4",
        "n_calibration_pairs": result["sample"]["n_total"],
        "auc_maintained": result["maintained_threshold"]["auc"],
        "auc_tracked":    result["tracked_threshold"]["auc"],
        "schema_version": "1.0",
    }
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Wrote calibration meta: %s", meta_path)


def load_calibrated_thresholds(result_path: Path) -> Dict[str, float]:
    """
    Read a calibration JSON and return ``{"maintained", "rephrased"}`` so the
    caller can pass it straight to ``SemanticContinuityMatcher``.

    NB04 Cell 34 in the original notebook does this inline; keeping it as a
    function makes the script side cleaner.
    """
    result_path = Path(result_path)
    with open(result_path) as fh:
        blob = json.load(fh)
    return {
        "maintained": float(blob["maintained_threshold"]["calibrated_f1"]),
        "rephrased":  float(blob["tracked_threshold"]["calibrated_f1"]),
    }
