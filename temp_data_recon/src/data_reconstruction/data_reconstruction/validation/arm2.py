"""Arm 2 orchestrator: FF5 + UMD reconstruction vs Ken French.

Per Doc 3 §"Arm 2", Arm 2 is the load-bearing gate for Stage 00.
Acceptance criteria (all factors, on the 2017-2024 overlap):

- Pearson rho > 0.85 vs Ken French
- MAD < 0.01 (1 percent monthly absolute)
- |annualized divergence| < 0.03 (3 percent per year absolute)
- No calendar-year correlation below 0.80

This module ties together `fama_french.build_ff_factors` (the
constructor on our panel) and `ken_french_loader.load_kf_factors`
(the published benchmark), then writes:

- `arm2_factor_returns.parquet`  --- both series side-by-side
- `arm2_per_factor_summary.parquet` --- per-factor rho / MAD / Delta_ann / per-year rho
- `arm2_summary.json` --- acceptance status + the four criteria
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_reconstruction.validation.fama_french import build_ff_factors
from data_reconstruction.validation.ken_french_loader import load_kf_factors

log = logging.getLogger(__name__)


# Map our column names to Ken French column names
FACTOR_PAIRS: list[tuple[str, str]] = [
    ("mkt_ours", "mktrf"),
    ("smb_ours", "smb"),
    ("hml_ours", "hml"),
    ("rmw_ours", "rmw"),
    ("cma_ours", "cma"),
    ("umd_ours", "umd"),
]


def _pearson(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def per_factor_diagnostics(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-factor rho, MAD, mean diff, annualized divergence, per-year rho."""
    rows: list[dict[str, Any]] = []
    joined = joined.copy()
    joined["year"] = joined["date"].dt.year

    for ours_col, kf_col in FACTOR_PAIRS:
        if ours_col not in joined.columns or kf_col not in joined.columns:
            continue
        sub = joined[["date", "year", ours_col, kf_col]].dropna()
        if len(sub) < 3:
            continue
        rho = _pearson(sub[ours_col], sub[kf_col])
        mad = float((sub[ours_col] - sub[kf_col]).abs().mean())
        mean_ours_ann = float(sub[ours_col].mean()) * 12.0
        mean_kf_ann = float(sub[kf_col].mean()) * 12.0
        delta_ann = mean_ours_ann - mean_kf_ann

        per_year_rho = (
            sub.groupby("year")
            .apply(lambda g: _pearson(g[ours_col], g[kf_col]), include_groups=False)
            .rename("rho")
        )
        min_year_rho = float(per_year_rho.min()) if len(per_year_rho) else float("nan")
        min_year = (
            int(per_year_rho.idxmin()) if len(per_year_rho) and pd.notna(per_year_rho.min()) else None
        )

        rows.append(
            {
                "factor": ours_col.replace("_ours", ""),
                "n_months": int(len(sub)),
                "pearson_rho": rho,
                "mad": mad,
                "ann_ours": mean_ours_ann,
                "ann_kf": mean_kf_ann,
                "delta_ann": delta_ann,
                "min_year_rho": min_year_rho,
                "min_year": min_year,
            }
        )
    return pd.DataFrame(rows)


def evaluate_acceptance(diag: pd.DataFrame) -> dict[str, Any]:
    """Apply the four Doc 3 §"Arm 2" criteria to per-factor diagnostics."""
    criteria: dict[str, Any] = {
        "pearson_gt_0p85": {
            "threshold": 0.85,
            "n_factors": int(len(diag)),
            "n_pass": int((diag["pearson_rho"] > 0.85).sum()),
            "failing": diag.loc[diag["pearson_rho"] <= 0.85, "factor"].tolist(),
        },
        "mad_lt_0p01": {
            "threshold": 0.01,
            "n_factors": int(len(diag)),
            "n_pass": int((diag["mad"] < 0.01).sum()),
            "failing": diag.loc[diag["mad"] >= 0.01, "factor"].tolist(),
        },
        "abs_delta_ann_lt_0p03": {
            "threshold": 0.03,
            "n_factors": int(len(diag)),
            "n_pass": int((diag["delta_ann"].abs() < 0.03).sum()),
            "failing": diag.loc[diag["delta_ann"].abs() >= 0.03, "factor"].tolist(),
        },
        "min_year_rho_gt_0p80": {
            "threshold": 0.80,
            "n_factors": int(len(diag)),
            "n_pass": int((diag["min_year_rho"] > 0.80).sum()),
            "failing": diag.loc[diag["min_year_rho"] <= 0.80, "factor"].tolist(),
        },
    }
    all_pass = all(
        c["n_pass"] == c["n_factors"] for c in criteria.values() if c["n_factors"] > 0
    )
    return {"all_passed": all_pass, "criteria": criteria}


def compare_to_kf(
    panel: pd.DataFrame,
    kf_factors: pd.DataFrame,
    foundation_dir: Path,
    *,
    sample_start: str = "2017-01-01",
    sample_end: str = "2024-12-31",
) -> dict[str, Any]:
    """Run Arm 2 end-to-end and write all artifacts.

    Parameters
    ----------
    panel
        Rank-normalized Stage 00 panel.
    kf_factors
        Ken French monthly factor returns (date, rf, mktrf, smb, hml, rmw, cma, umd).
    foundation_dir
        Directory for ``arm2_*.parquet`` outputs and ``arm2_summary.json``.
    """
    foundation_dir = Path(foundation_dir)
    foundation_dir.mkdir(parents=True, exist_ok=True)

    ours_ff = build_ff_factors(panel, sample_start=sample_start, sample_end=sample_end)
    kf = kf_factors.copy()
    kf["date"] = pd.to_datetime(kf["date"]) + pd.offsets.MonthEnd(0)
    kf = kf[(kf["date"] >= sample_start) & (kf["date"] <= sample_end)]

    joined = ours_ff.merge(kf, on="date", how="inner")
    joined.to_parquet(foundation_dir / "arm2_factor_returns.parquet", index=False)
    log.info("arm2: joined factor returns %d months -> arm2_factor_returns.parquet", len(joined))

    diag = per_factor_diagnostics(joined)
    diag.to_parquet(foundation_dir / "arm2_per_factor_summary.parquet", index=False)
    for _, row in diag.iterrows():
        log.info(
            "arm2.%s: rho=%.3f mad=%.4f delta_ann=%+.4f min_year_rho=%.3f (year=%s)",
            row["factor"],
            row["pearson_rho"],
            row["mad"],
            row["delta_ann"],
            row["min_year_rho"],
            row["min_year"],
        )

    acceptance = evaluate_acceptance(diag)
    summary: dict[str, Any] = {
        "arm2_severity": "hard",
        "arm2_passed": bool(acceptance["all_passed"]),
        "sample_window": [sample_start, sample_end],
        "n_months_compared": int(len(joined)),
        "acceptance": acceptance,
        "diagnostics": diag.to_dict(orient="records"),
    }
    out_path = foundation_dir / "arm2_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("arm2: summary -> %s", out_path)
    log.log(
        logging.INFO if summary["arm2_passed"] else logging.ERROR,
        "arm2: ACCEPTANCE %s",
        "PASS" if summary["arm2_passed"] else "FAIL",
    )
    return summary
