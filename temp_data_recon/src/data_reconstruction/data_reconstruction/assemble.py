"""Final panel assembly, completeness filter, and rank normalization.

Stage 00 produces both the canonical panel (`factor_panel_v2.parquet`)
and the train/valid/test splits per the project's documented period
convention (pre-sample 1967-1971, training 1972-2010, OOS 2011-2021,
holdout 2022-2024 --- see `COLLABORATOR_PROMPT.txt` §"Known prior
decisions"). The 1967-1971 pre-sample rows remain in `factor_panel_v2`
so downstream stages can use them as lookback for momentum / long-horizon
windows; they are not assigned to any split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_reconstruction.config import Stage00Config
from data_reconstruction.constants import ALL_46

log = logging.getLogger(__name__)


def compute_me_dependent_characteristics(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute characteristics whose denominator is current market equity."""
    panel = panel.copy()
    panel["BEME"] = (panel["BE_raw"] / panel["me"]).where(panel["me"] > 0)
    panel["E2P"] = (panel["IB_raw"] / panel["me"]).where(panel["me"] > 0)
    panel["CF2P"] = (panel["CF_raw"] / panel["me"]).where(panel["me"] > 0)
    panel["D2P"] = (panel["DV_raw"].fillna(0.0) / panel["me"]).where(panel["me"] > 0)
    panel["S2P"] = (panel["SALE_raw"] / panel["me"]).where(panel["me"] > 0)
    panel["A2ME"] = (panel["AT_raw"] / panel["me"]).where(panel["me"] > 0)
    panel["Q"] = ((panel["AT_for_Q"] + panel["me"] - panel["BE_for_Q"].fillna(0.0)) / panel["AT_for_Q"]).where(
        (panel["AT_for_Q"] > 0) & (panel["me"] > 0)
    )
    panel["CF"] = (panel["CF_raw"] / panel["me"]).where(panel["me"] > 0)
    debt_plus_me = panel["DEBT_raw"] + panel["me"]
    panel["Lev"] = (panel["DEBT_raw"] / debt_plus_me).where(debt_plus_me > 0)
    return panel


def write_completeness_diagnostics(panel: pd.DataFrame, char_cols: list[str], out_dir: Path) -> pd.DataFrame:
    """Write pre-filter non-null counts for every characteristic.

    Also writes a per-(year, characteristic) coverage table to
    `yearly_completeness_diag.parquet`, used by Arm 1 to plot a coverage
    heatmap that explains universe-size differences vs CPZ.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = pd.DataFrame(
        {
            "characteristic": char_cols,
            "non_null": [int(panel[c].notna().sum()) for c in char_cols],
            "rows": len(panel),
        }
    )
    diag["pct_present"] = diag["non_null"] / diag["rows"].where(diag["rows"] != 0, np.nan)
    diag.to_csv(out_dir / "completeness_diag.csv", index=False)

    year = pd.to_datetime(panel["date"]).dt.year
    rows: list[dict[str, object]] = []
    for col in char_cols:
        cov_by_year = panel.assign(_year=year).groupby("_year")[col].apply(
            lambda s: s.notna().mean()
        )
        for yr, c in cov_by_year.items():
            rows.append({"year": int(yr), "characteristic": col, "coverage": float(c)})
    pd.DataFrame(rows).to_parquet(out_dir / "yearly_completeness_diag.parquet", index=False)

    return diag


def apply_completeness_filter(panel: pd.DataFrame, char_cols: list[str]) -> pd.DataFrame:
    """Keep only firm-months with all characteristics present."""
    return panel[panel[char_cols].notna().all(axis=1)].copy()


def rank_normalize(series: pd.Series) -> pd.Series:
    """Cross-sectional rank normalization to the open interval (-0.5, 0.5)."""
    n = series.notna().sum()
    if n < 2:
        return series
    ranks = series.rank(method="average", na_option="keep")
    return ranks / (n + 1.0) - 0.5


def rank_normalize_panel(panel: pd.DataFrame, char_cols: list[str]) -> pd.DataFrame:
    """Rank-normalize characteristics within each month."""
    panel = panel.sort_values("date").reset_index(drop=True).copy()
    chunks = []
    for _, year_data in panel.groupby(panel["date"].dt.year, sort=True):
        part = year_data.copy()
        part[char_cols] = part.groupby("date")[char_cols].transform(rank_normalize)
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True) if chunks else panel


def assemble_panel(config: Stage00Config) -> pd.DataFrame:
    """Assemble, filter, rank-normalize, and save the final Stage 00 panel."""
    config.ensure_dirs()
    log.info("assemble: reading monthly/risk/accounting intermediates from %s", config.raw_dir)
    monthly = pd.read_parquet(config.raw_dir / "monthly_chars.parquet")
    risk = pd.read_parquet(config.raw_dir / "risk_chars.parquet")
    accounting = pd.read_parquet(config.raw_dir / "accounting_chars.parquet")

    for df in [monthly, risk]:
        df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
        df["permno"] = df["permno"].astype("int64")
    accounting["avail_date"] = pd.to_datetime(accounting["avail_date"]) + pd.offsets.MonthEnd(0)
    accounting["permno"] = accounting["permno"].astype("int64")

    panel = monthly[
        (monthly["date"] >= config.sample_start)
        & (monthly["date"] <= config.sample_end)
        & monthly["ret_excess"].notna()
    ].copy()
    log.info(
        "assemble: monthly rows in sample window %s..%s: %d",
        config.sample_start,
        config.sample_end,
        len(panel),
    )

    panel = panel.merge(
        risk[["permno", "date", "Beta", "MktBeta", "IdioVol", "Resid_Var", "Variance", "Spread"]],
        on=["permno", "date"],
        how="left",
    )

    panel = panel.sort_values("date").reset_index(drop=True)
    acc_slim = accounting.rename(columns={"avail_date": "date"}).sort_values("date").reset_index(drop=True)
    panel = pd.merge_asof(
        panel,
        acc_slim,
        on="date",
        by="permno",
        direction="backward",
        tolerance=pd.Timedelta(days=config.pit_tolerance_days),
    )
    log.info(
        "assemble: merge_asof with tolerance=%d days; rows post-merge: %d",
        config.pit_tolerance_days,
        len(panel),
    )
    panel = compute_me_dependent_characteristics(panel)

    missing = [c for c in ALL_46 if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing characteristic columns: {missing}")

    diag = write_completeness_diagnostics(panel, ALL_46, config.foundation_dir)
    filtered = apply_completeness_filter(panel, ALL_46)
    log.info(
        "assemble: completeness filter %d -> %d rows (%.1f%% retained)",
        len(panel),
        len(filtered),
        100.0 * len(filtered) / max(len(panel), 1),
    )
    ranked = rank_normalize_panel(filtered, ALL_46)

    panel_path = config.foundation_dir / "factor_panel_v2.parquet"
    ranked.to_parquet(panel_path, index=False)
    log.info("assemble: final panel %s rows -> %s", f"{len(ranked):,}", panel_path)

    split_counts = _save_splits(ranked, config.output_dir)

    summary = {
        "rows_before_filter": int(len(panel)),
        "rows_after_filter": int(len(filtered)),
        "rows_final": int(len(ranked)),
        "n_firms": int(ranked["permno"].nunique()) if len(ranked) else 0,
        "date_min": str(ranked["date"].min().date()) if len(ranked) else None,
        "date_max": str(ranked["date"].max().date()) if len(ranked) else None,
        "all_zero_characteristics": diag.loc[diag["non_null"] == 0, "characteristic"].tolist(),
        "splits": split_counts,
    }
    (config.foundation_dir / "stage00_summary.json").write_text(json.dumps(summary, indent=2))
    return ranked


#: Documented period boundaries. Pre-sample 1967-1971 is retained in
#: `factor_panel_v2.parquet` for lookback but is not assigned to any split.
SPLIT_BOUNDARIES = {
    "train": ("1972-01-01", "2010-12-31"),
    "valid": ("2011-01-01", "2021-12-31"),
    "test":  ("2022-01-01", "2024-12-31"),
}


def _save_splits(panel: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    """Write train/valid/test splits per `SPLIT_BOUNDARIES`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, (lo, hi) in SPLIT_BOUNDARIES.items():
        part = panel[(panel["date"] >= lo) & (panel["date"] <= hi)].copy()
        out = output_dir / f"{name}.parquet"
        part.to_parquet(out, index=False)
        counts[name] = int(len(part))
        log.info("assemble: split %s [%s..%s] -> %d rows -> %s", name, lo, hi, len(part), out)
    return counts

