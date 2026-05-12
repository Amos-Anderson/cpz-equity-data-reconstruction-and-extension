"""Daily risk and liquidity characteristic construction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from data_reconstruction.config import Stage00Config

log = logging.getLogger(__name__)


def compute_risk_characteristics(dsf: pd.DataFrame, ff_daily: pd.DataFrame, target_months: pd.DataFrame) -> pd.DataFrame:
    """Compute Beta, IdioVol, Resid_Var, and Variance using 252 daily obs."""
    dsf = dsf.copy()
    ff_daily = ff_daily.copy()
    dsf["date"] = pd.to_datetime(dsf["date"])
    ff_daily["date"] = pd.to_datetime(ff_daily["date"])
    dsf["permno"] = dsf["permno"].astype("int64")
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce").astype("float64")
    for col in ["mktrf", "rf"]:
        ff_daily[col] = pd.to_numeric(ff_daily[col], errors="coerce").astype("float64")

    dsf = dsf.merge(ff_daily[["date", "mktrf", "rf"]], on="date", how="left")
    dsf["ret_excess_d"] = dsf["ret"] - dsf["rf"]
    dsf = dsf.sort_values(["permno", "date"]).reset_index(drop=True)

    target_months = target_months.copy()
    target_months["date"] = pd.to_datetime(target_months["date"]) + pd.offsets.MonthEnd(0)
    target_months["permno"] = target_months["permno"].astype("int64")

    results: list[dict[str, object]] = []
    target_by_permno = {p: g["date"].to_numpy() for p, g in target_months.groupby("permno")}
    for permno, pdata in dsf.groupby("permno", sort=False):
        months = target_by_permno.get(permno)
        if months is None or len(pdata) < 60:
            continue
        pdata = pdata.sort_values("date")
        ret_d = pdata["ret_excess_d"].to_numpy(dtype="float64")
        mkt_d = pdata["mktrf"].to_numpy(dtype="float64")
        dates_d = pdata["date"].to_numpy()

        for month_end in months:
            idx_end = np.searchsorted(dates_d, np.datetime64(month_end), side="right")
            idx_start = max(0, idx_end - 252)
            if idx_end - idx_start < 60:
                continue
            r = ret_d[idx_start:idx_end]
            m = mkt_d[idx_start:idx_end]
            valid = np.isfinite(r) & np.isfinite(m)
            if valid.sum() < 60:
                continue
            r_v = r[valid]
            m_v = m[valid]
            try:
                slope, intercept, _, _, _ = stats.linregress(m_v, r_v)
            except ValueError:
                continue
            resid = r_v - (intercept + slope * m_v)
            results.append(
                {
                    "permno": permno,
                    "date": pd.Timestamp(month_end) + pd.offsets.MonthEnd(0),
                    "Beta": slope,
                    "MktBeta": slope,
                    "IdioVol": np.std(resid, ddof=1) * np.sqrt(252),
                    "Resid_Var": np.var(resid, ddof=1) * 252,
                    "Variance": np.var(r_v, ddof=1) * 252,
                }
            )
    return pd.DataFrame(results)


def compute_roll_spread(dsf: pd.DataFrame) -> pd.DataFrame:
    """Compute Roll (1984) implied spread by stock-month."""
    dsf = dsf.copy()
    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["permno"] = dsf["permno"].astype("int64")
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce").astype("float64")
    dsf["ym"] = dsf["date"].dt.to_period("M")

    def _spread(ret: pd.Series) -> float:
        r = ret.dropna().to_numpy(dtype="float64")
        if len(r) < 10:
            return np.nan
        cov = np.cov(r[1:], r[:-1], bias=True)[0, 1]
        return float(2.0 * np.sqrt(-cov)) if cov < 0 else 0.0

    out = dsf.groupby(["permno", "ym"], sort=False)["ret"].apply(_spread).rename("Spread")
    out = out.reset_index()
    out["date"] = out["ym"].dt.to_timestamp("M") + pd.offsets.MonthEnd(0)
    return out[["permno", "date", "Spread"]]


def build_risk_file(config: Stage00Config) -> pd.DataFrame:
    """Read daily/monthly intermediates, compute risk and spread, and save."""
    log.info("risk: reading daily CRSP + FF daily + monthly targets from %s", config.raw_dir)
    dsf = pd.read_parquet(config.raw_dir / "crsp_dsf_raw.parquet")
    ff_daily = pd.read_parquet(config.raw_dir / "ff_factors_daily.parquet")
    monthly = pd.read_parquet(config.raw_dir / "monthly_chars.parquet")
    target_months = monthly[
        (pd.to_datetime(monthly["date"]) >= "1966-01-31")
        & (pd.to_datetime(monthly["date"]) <= config.sample_end)
    ][["permno", "date"]]
    log.info(
        "risk: dsf rows=%d, ff_daily rows=%d, target firm-months=%d; running 252d CAPM (slow)",
        len(dsf),
        len(ff_daily),
        len(target_months),
    )

    risk = compute_risk_characteristics(dsf, ff_daily, target_months)
    log.info("risk: CAPM regressions complete; computing Roll spread")
    spread = compute_roll_spread(dsf)
    risk = risk.merge(spread, on=["permno", "date"], how="left")
    out_path = config.raw_dir / "risk_chars.parquet"
    risk.to_parquet(out_path, index=False)
    log.info("risk: %d firm-months -> %s", len(risk), out_path)
    return risk
