"""Clean CRSP monthly returns and compute market equity."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_reconstruction.config import Stage00Config

log = logging.getLogger(__name__)


def _to_float(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")


def build_clean_monthly(config: Stage00Config) -> pd.DataFrame:
    """Build `crsp_clean_monthly.parquet` from raw MSF, delistings, and FF RF.

    Returns
    -------
    pandas.DataFrame
        CRSP monthly panel with `ret_adj`, `ret_excess`, and `me`.
    """
    raw_dir = config.raw_dir
    log.info("crsp: reading raw MSF, delist, FF monthly from %s", raw_dir)
    msf = pd.read_parquet(raw_dir / "crsp_msf_raw.parquet")
    delist = pd.read_parquet(raw_dir / "crsp_delist_raw.parquet")
    ff = pd.read_parquet(raw_dir / "ff_factors_monthly_full.parquet")
    log.info("crsp: MSF rows=%d, delist rows=%d, FF months=%d", len(msf), len(delist), len(ff))

    msf["date"] = pd.to_datetime(msf["date"]) + pd.offsets.MonthEnd(0)
    delist["date"] = pd.to_datetime(delist["date"]) + pd.offsets.MonthEnd(0)
    ff["date"] = pd.to_datetime(ff["date"]) + pd.offsets.MonthEnd(0)

    _to_float(msf, ["ret", "retx", "prc", "shrout", "vol", "cfacpr", "cfacshr"])
    _to_float(ff, ["rf", "mktrf"])
    _to_float(delist, ["dlret", "dlstcd"])
    msf["permno"] = msf["permno"].astype("int64")
    delist["permno"] = delist["permno"].astype("int64")

    merged = msf.merge(delist[["permno", "date", "dlret", "dlstcd"]], on=["permno", "date"], how="left")
    perf = merged["dlstcd"].between(520, 584) | (merged["dlstcd"] == 500)
    missing_dlret = merged["dlret"].isna()
    nyse_amex = merged["exchcd"].isin([1, 2])
    nasdaq = merged["exchcd"] == 3

    merged["dlret_fill"] = merged["dlret"]
    merged.loc[perf & missing_dlret & nyse_amex, "dlret_fill"] = config.nyse_amex_delist_return
    merged.loc[perf & missing_dlret & nasdaq, "dlret_fill"] = config.nasdaq_delist_return
    merged.loc[merged["dlstcd"].notna() & ~perf & missing_dlret, "dlret_fill"] = 0.0

    merged["ret_adj"] = merged["ret"]
    has_dl = merged["dlret_fill"].notna()
    both = has_dl & merged["ret"].notna()
    only_dl = has_dl & merged["ret"].isna()
    merged.loc[both, "ret_adj"] = (
        (1.0 + merged.loc[both, "ret"]) * (1.0 + merged.loc[both, "dlret_fill"]) - 1.0
    )
    merged.loc[only_dl, "ret_adj"] = merged.loc[only_dl, "dlret_fill"]

    merged = merged.merge(ff[["date", "rf", "mktrf"]], on="date", how="left")
    merged["ret_excess"] = merged["ret_adj"] - merged["rf"]
    merged["me"] = merged["prc"].abs() * merged["shrout"] / 1000.0
    merged["me"] = merged["me"].where(merged["me"] > 0)

    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)
    n_perf_imputed = int((perf & missing_dlret & (nyse_amex | nasdaq)).sum())
    out_path = raw_dir / "crsp_clean_monthly.parquet"
    merged.to_parquet(out_path, index=False)
    log.info(
        "crsp: %d rows; Shumway-imputed delisting returns: %d; -> %s",
        len(merged),
        n_perf_imputed,
        out_path,
    )
    return merged

