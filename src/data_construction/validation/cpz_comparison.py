"""Arm 1: distributional comparison of our Stage 00 panel vs CPZ (1967-2016).

The CPZ reference panel distributed by Chen-Pelger-Zhu (via Stefan Jansen's
replication archive) is permno-anonymized at source --- 48 columns (date,
ret, 46 rank-normalized characteristics) with an integer row index and no
firm identifier. This is a licensing artifact (CRSP/Compustat restrictions
on identifiable redistribution), not a missing-file issue.

Without firm identifiers, the per-(permno, date) intersection comparison
prescribed in the original Doc 3 cannot be implemented. We instead compute
the four *distributional* artifacts that the previous notebook
(`Previous Data Pipeline Versions/Data_Pipeline_v2.ipynb`) used:

1. ``arm1_annual_breadth.parquet``  --- yearly mean firms-per-month.
2. ``arm1_annual_returns.parquet``  --- yearly mean excess return,
   equal-weighted.
3. ``arm1_char_stats.parquet``      --- per-characteristic full-period
   mean / median / std on the rank-normalized scale.
4. ``arm1_yearly_coverage.parquet`` --- per-(year, char) pre-filter
   coverage of our panel; no CPZ analog (CPZ is published post-filter
   so its coverage is trivially 1.0).

Arm 1 is reported as a *diagnostic layer*. The acceptance gate for
Stage 00 sits at Arms 2 (FF5+UMD replication) and 3 (internal audits).

There is no lookahead bias: all comparisons are restricted to
1967-2016 (the CPZ panel's coverage); each metric is computed
within the month (no cross-month leakage); the rank-normalization
that produces the inputs is itself cross-sectional per month.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_reconstruction.constants import ALL_46

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# CPZ panel loading
# -----------------------------------------------------------------------------


def load_cpz_panel(cpz_path: Path) -> pd.DataFrame:
    """Load and normalize the CPZ reference panel.

    Returns a DataFrame with `date` snapped to month-end (one row per
    firm-month observation; rows are not joinable to ours because the
    CPZ panel carries no firm identifier).
    """
    cpz = pd.read_parquet(cpz_path)
    cpz = cpz.copy()
    cpz["date"] = pd.to_datetime(cpz["date"]) + pd.offsets.MonthEnd(0)
    return cpz


# -----------------------------------------------------------------------------
# Artifact 1 --- Annual breadth
# -----------------------------------------------------------------------------


def annual_breadth_table(ours: pd.DataFrame, cpz: pd.DataFrame) -> pd.DataFrame:
    """Per-year mean firms-per-month, ours vs CPZ, with diff and pct_diff."""

    def _breadth(df: pd.DataFrame) -> pd.Series:
        year = df["date"].dt.year
        per_year = df.groupby(year).size() / df.groupby(year)["date"].nunique()
        return per_year

    ours_b = _breadth(ours).rename("ours_breadth").rename_axis("year")
    cpz_b = _breadth(cpz).rename("cpz_breadth").rename_axis("year")
    out = pd.concat([ours_b, cpz_b], axis=1).reset_index().sort_values("year")
    out["diff"] = out["ours_breadth"] - out["cpz_breadth"]
    out["pct_diff"] = 100.0 * out["diff"] / out["cpz_breadth"]
    return out


# -----------------------------------------------------------------------------
# Artifact 2 --- Annual returns
# -----------------------------------------------------------------------------


def annual_returns_table(ours: pd.DataFrame, cpz: pd.DataFrame) -> pd.DataFrame:
    """Per-year equal-weighted excess return, ours vs CPZ.

    CPZ's `ret` column is already excess return per the CPZ data dictionary
    (the dataset stores excess returns, not raw returns). Our side uses
    `ret_excess` from the assembled panel.
    """
    ours_ar = ours.groupby(ours["date"].dt.year)["ret_excess"].mean().rename("ours_ann_ret")
    cpz_ar = cpz.groupby(cpz["date"].dt.year)["ret"].mean().rename("cpz_ann_ret")
    out = (
        pd.concat([ours_ar, cpz_ar], axis=1)
        .rename_axis("year")
        .reset_index()
        .sort_values("year")
    )
    out["diff"] = out["ours_ann_ret"] - out["cpz_ann_ret"]
    return out


# -----------------------------------------------------------------------------
# Artifact 3 --- Per-characteristic full-period stats
# -----------------------------------------------------------------------------


def char_stats_table(ours: pd.DataFrame, cpz: pd.DataFrame, char_cols: list[str]) -> pd.DataFrame:
    """Per-characteristic full-period mean / median / std (rank-normalized)."""
    rows: list[dict[str, Any]] = []
    for col in char_cols:
        if col not in ours.columns or col not in cpz.columns:
            continue
        o = ours[col].dropna()
        c = cpz[col].dropna()
        rows.append(
            {
                "characteristic": col,
                "ours_mean": float(o.mean()) if len(o) else np.nan,
                "ours_median": float(o.median()) if len(o) else np.nan,
                "ours_std": float(o.std()) if len(o) else np.nan,
                "cpz_mean": float(c.mean()) if len(c) else np.nan,
                "cpz_median": float(c.median()) if len(c) else np.nan,
                "cpz_std": float(c.std()) if len(c) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["diff_mean"] = out["ours_mean"] - out["cpz_mean"]
    out["diff_median"] = out["ours_median"] - out["cpz_median"]
    out["diff_std"] = out["ours_std"] - out["cpz_std"]
    return out


# -----------------------------------------------------------------------------
# Artifact 4 --- Per-(year, char) pre-filter coverage diagnostic
# -----------------------------------------------------------------------------


def load_yearly_coverage(foundation_dir: Path) -> pd.DataFrame:
    """Read the pre-filter per-(year, char) coverage table written by `assemble`."""
    p = foundation_dir / "yearly_completeness_diag.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["year", "characteristic", "coverage"])
    return pd.read_parquet(p)


# -----------------------------------------------------------------------------
# Top-level orchestrator
# -----------------------------------------------------------------------------


def compare_to_cpz(
    ours_panel: pd.DataFrame,
    cpz_path: Path,
    foundation_dir: Path,
    *,
    sample_start: str = "1967-01-01",
    sample_end: str = "2016-12-31",
    char_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Run Arm 1 against the CPZ reference panel and write all four artifacts.

    Parameters
    ----------
    ours_panel
        Our rank-normalized Stage 00 panel.
    cpz_path
        Path to the CPZ reference parquet (`firm_characteristics_all.parquet`).
    foundation_dir
        Directory for `arm1_*.parquet` outputs and `arm1_summary.json`.
    sample_start, sample_end
        Restrict both panels to this window (CPZ ends in 2016).
    char_cols
        Characteristic columns to include. Defaults to `ALL_46`.

    Returns
    -------
    dict
        Summary written to `arm1_summary.json`.
    """
    foundation_dir = Path(foundation_dir)
    foundation_dir.mkdir(parents=True, exist_ok=True)
    cols = char_cols or ALL_46

    if not Path(cpz_path).exists():
        log.warning("Arm 1 skipped: CPZ reference not found at %s", cpz_path)
        return {"arm1_run": False, "reason": f"CPZ reference missing at {cpz_path}"}

    cpz = load_cpz_panel(Path(cpz_path))
    ours = ours_panel.copy()
    ours["date"] = pd.to_datetime(ours["date"]) + pd.offsets.MonthEnd(0)

    ours = ours[(ours["date"] >= sample_start) & (ours["date"] <= sample_end)]
    cpz = cpz[(cpz["date"] >= sample_start) & (cpz["date"] <= sample_end)]
    log.info(
        "arm1: comparing on %s..%s; ours rows=%d, cpz rows=%d",
        sample_start,
        sample_end,
        len(ours),
        len(cpz),
    )

    # Artifact 1
    breadth = annual_breadth_table(ours, cpz)
    breadth.to_parquet(foundation_dir / "arm1_annual_breadth.parquet", index=False)
    log.info(
        "arm1.breadth: %d years; max |pct_diff| = %.1f%%",
        len(breadth),
        breadth["pct_diff"].abs().max(),
    )

    # Artifact 2
    returns = annual_returns_table(ours, cpz)
    returns.to_parquet(foundation_dir / "arm1_annual_returns.parquet", index=False)
    log.info(
        "arm1.returns: %d years; max |diff| = %.4f",
        len(returns),
        returns["diff"].abs().max(),
    )

    # Artifact 3
    stats = char_stats_table(ours, cpz, cols)
    stats.to_parquet(foundation_dir / "arm1_char_stats.parquet", index=False)
    log.info(
        "arm1.char_stats: %d chars; max |diff_mean| = %.4f, max |diff_std| = %.4f",
        len(stats),
        stats["diff_mean"].abs().max(),
        stats["diff_std"].abs().max(),
    )

    # Artifact 4: copy the pre-filter yearly coverage written by `assemble`
    # into a stable arm1-named location, restricted to the CPZ window.
    cov = load_yearly_coverage(foundation_dir)
    if len(cov):
        cov = cov[(cov["year"] >= 1967) & (cov["year"] <= 2016)].copy()
        cov.to_parquet(foundation_dir / "arm1_yearly_coverage.parquet", index=False)
        log.info("arm1.coverage: %d (year, char) cells written", len(cov))

    summary: dict[str, Any] = {
        "arm1_run": True,
        "arm1_severity": "diagnostic",
        "sample_window": [sample_start, sample_end],
        "ours_rows": int(len(ours)),
        "cpz_rows": int(len(cpz)),
        "breadth": {
            "n_years": int(len(breadth)),
            "years_pct_diff_within_10": int((breadth["pct_diff"].abs() <= 10).sum()),
            "years_pct_diff_within_20": int((breadth["pct_diff"].abs() <= 20).sum()),
            "max_abs_pct_diff": float(breadth["pct_diff"].abs().max()),
            "max_abs_pct_diff_year": int(
                breadth.loc[breadth["pct_diff"].abs().idxmax(), "year"]
            )
            if len(breadth)
            else None,
        },
        "returns": {
            "n_years": int(len(returns)),
            "years_within_0p5pct": int((returns["diff"].abs() < 0.005).sum()),
            "max_abs_diff": float(returns["diff"].abs().max()),
            "max_abs_diff_year": int(returns.loc[returns["diff"].abs().idxmax(), "year"])
            if len(returns)
            else None,
        },
        "char_stats": {
            "n_chars": int(len(stats)),
            "max_abs_diff_mean": float(stats["diff_mean"].abs().max()),
            "max_abs_diff_std": float(stats["diff_std"].abs().max()),
            "char_with_max_diff_mean": str(
                stats.loc[stats["diff_mean"].abs().idxmax(), "characteristic"]
            )
            if len(stats)
            else None,
        },
        "coverage": {
            "n_cells": int(len(cov)),
            "min_coverage": float(cov["coverage"].min()) if len(cov) else None,
            "n_cells_below_0p80": int((cov["coverage"] < 0.80).sum()) if len(cov) else 0,
        },
    }
    out_json = foundation_dir / "arm1_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("arm1 summary -> %s", out_json)
    return summary
