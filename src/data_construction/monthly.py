"""Monthly CRSP-based characteristic construction."""

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


def _compound_from_log(log_returns: pd.Series, skip: int, window: int, min_periods: int) -> pd.Series:
    log_sum = log_returns.shift(skip).rolling(window, min_periods=min_periods).sum()
    return np.exp(log_sum) - 1.0


def compute_suv(group: pd.DataFrame) -> pd.Series:
    """Standardized unexplained volume from an AR(3) log-volume model."""
    lv = group["log_vol"].to_numpy(dtype="float64")
    n = len(lv)
    suv = np.full(n, np.nan)
    for i in range(12, n):
        window = min(i, 36)
        min_len = min(window, i - 3)
        if min_len < 12:
            continue
        y = lv[i - min_len : i]
        x1 = lv[i - min_len - 1 : i - 1]
        x2 = lv[i - min_len - 2 : i - 2]
        x3 = lv[i - min_len - 3 : i - 3]
        m = min(len(y), len(x1), len(x2), len(x3))
        if m < 12:
            continue
        y = y[-m:]
        x = np.column_stack([np.ones(m), x1[-m:], x2[-m:], x3[-m:]])
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if valid.sum() < 12:
            continue
        try:
            coef, _, _, _ = np.linalg.lstsq(x[valid], y[valid], rcond=None)
            residuals = y[valid] - x[valid] @ coef
            std_r = np.std(residuals, ddof=4)
            if std_r > 0 and np.isfinite(lv[i]):
                suv[i] = (lv[i] - x[-1] @ coef) / std_r
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pd.Series(suv, index=group.index)


def compute_monthly_characteristics(msf: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly characteristics from clean CRSP monthly data."""
    msf = msf.copy()
    msf["date"] = pd.to_datetime(msf["date"]) + pd.offsets.MonthEnd(0)
    msf["permno"] = msf["permno"].astype("int64")
    _to_float(msf, ["ret", "ret_adj", "ret_excess", "me", "prc", "shrout", "vol", "cfacpr", "cfacshr"])
    msf = msf.sort_values(["permno", "date"]).reset_index(drop=True)
    group = msf.groupby("permno", group_keys=False)

    msf["LME"] = np.log(group["me"].shift(1).clip(lower=1e-6))
    turnover = msf["vol"] / (msf["shrout"] * 1000.0).clip(lower=1.0)
    msf["LTurnover"] = np.log(turnover.clip(lower=1e-8))

    msf["log_ret_raw"] = np.log((1.0 + msf["ret_adj"]).clip(lower=0.01))
    msf["r2_1"] = group["ret_adj"].shift(1)
    msf["ST_REV"] = -msf["r2_1"]
    msf["r12_2"] = group["log_ret_raw"].transform(lambda x: _compound_from_log(x, 2, 11, 8))
    msf["r12_7"] = group["log_ret_raw"].transform(lambda x: _compound_from_log(x, 7, 6, 4))
    msf["r36_13"] = group["log_ret_raw"].transform(lambda x: _compound_from_log(x, 13, 24, 16))
    msf["LT_Rev"] = group["log_ret_raw"].transform(lambda x: _compound_from_log(x, 13, 48, 24))

    msf["prc_adj"] = msf["prc"].abs() / msf["cfacpr"].replace(0, np.nan)
    msf["Rel2High"] = msf["prc_adj"] / group["prc_adj"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=8).max()
    )

    msf["adj_shrout"] = msf["shrout"] * msf["cfacshr"].fillna(1.0)
    msf["log_adj_shrout"] = np.log(msf["adj_shrout"].clip(lower=1e-6))
    msf["NI"] = group["log_adj_shrout"].transform(lambda x: x - x.shift(12))

    msf["log_vol"] = np.log(msf["vol"].clip(lower=1e-6))
    msf["SUV"] = group.apply(compute_suv)

    monthly_cols = [
        "permno",
        "date",
        "me",
        "ret_adj",
        "ret_excess",
        "rf",
        "mktrf",
        "prc",
        "shrout",
        "vol",
        "cfacshr",
        "cfacpr",
        "shrcd",
        "exchcd",
        "siccd",
        "LME",
        "LTurnover",
        "r2_1",
        "r12_2",
        "r12_7",
        "r36_13",
        "LT_Rev",
        "ST_REV",
        "Rel2High",
        "NI",
        "SUV",
    ]
    return msf[[c for c in monthly_cols if c in msf.columns]]


def build_monthly_file(config: Stage00Config) -> pd.DataFrame:
    """Read clean CRSP monthly data, compute monthly characteristics, and save."""
    log.info("monthly: reading clean CRSP monthly from %s", config.raw_dir)
    msf = pd.read_parquet(config.raw_dir / "crsp_clean_monthly.parquet")
    log.info("monthly: %d firm-months in; computing momentum, NI, Rel2High, SUV (slow)", len(msf))
    monthly = compute_monthly_characteristics(msf)
    out_path = config.raw_dir / "monthly_chars.parquet"
    monthly.to_parquet(out_path, index=False)
    log.info(
        "monthly: SUV defined %d / %d (%.1f%%); -> %s",
        int(monthly["SUV"].notna().sum()),
        len(monthly),
        100.0 * monthly["SUV"].notna().mean(),
        out_path,
    )
    return monthly

