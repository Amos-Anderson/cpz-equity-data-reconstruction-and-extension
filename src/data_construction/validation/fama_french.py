"""FF5 + UMD long-short factor construction from our Stage 00 panel.

Implementation follows Doc 3 §"Arm 2" --- Factor construction protocol:

- Universe at each rebalance date: panel firms with non-missing
  `me`, the focal characteristic, and `exchcd` in {1, 2, 3}.
- NYSE-only size breakpoint (median of `exchcd == 1` ME).
- NYSE-only characteristic 30 / 70 percentile breakpoints.
- 2 (size) x 3 (char) sort → 6 portfolios.
- VW returns within each portfolio, weights proportional to `me_lag1`
  (previous month-end ME).
- Annual rebalance at June-end for HML, RMW, CMA, SMB; held July(y)
  through June(y+1).
- Monthly rebalance for UMD.
- MKT_RMRF = VW excess return of the universe.

Sign conventions:

- HML = (high BEME) - (low BEME)
- RMW = (high OP / Robust) - (low OP / Weak)
- CMA = (low Investment / Conservative) - (high Investment / Aggressive)  [REVERSED]
- UMD = (high r12_2 / Up / Winners) - (low r12_2 / Down / Losers)
- SMB = (Small) - (Big) averaged over BEME tertiles (Doc 3 single-sort
  variant; not the FF5 three-sort SMB).
- MKT = VW(ret_excess) of the universe.

Caveat: our panel has financial firms excluded
(`exclude_financials = True` in `Stage00Config`); Ken French's
portfolios include financial firms. This will create a small but
non-zero correlation gap with KF. Documented as a known divergence.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Columns we need on the input panel
_REQUIRED_COLS = ("permno", "date", "exchcd", "me", "ret_excess", "BEME", "OP", "Investment", "r12_2")


def _validate_panel(panel: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing required columns for FF construction: {missing}")


def _june_endpoints(years: Iterable[int]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(f"{y}-06-30") + pd.offsets.MonthEnd(0) for y in years])


def _assign_buckets(
    snapshot: pd.DataFrame,
    char_col: str,
    *,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> pd.DataFrame:
    """Add (size_bucket, char_bucket) columns to a per-rebalance snapshot.

    Uses NYSE-only median for size and NYSE-only 30 / 70 percentiles for
    the characteristic. Firms with NaN in `me` or `char_col` are dropped.
    """
    s = snapshot.dropna(subset=["me", char_col, "exchcd"]).copy()
    nyse = s[s["exchcd"] == 1]
    if len(nyse) < 30:  # cross-section too small to be meaningful
        return pd.DataFrame()

    me_median = float(nyse["me"].median())
    p30 = float(nyse[char_col].quantile(0.30))
    p70 = float(nyse[char_col].quantile(0.70))

    s["size_bucket"] = np.where(s["me"] <= me_median, "S", "B")
    s["char_bucket"] = pd.cut(
        s[char_col],
        bins=[-np.inf, p30, p70, np.inf],
        labels=[low_label, mid_label, high_label],
    ).astype(object)
    s = s.dropna(subset=["char_bucket"])
    return s[["permno", "size_bucket", "char_bucket"]]


def _vw_return(group: pd.DataFrame) -> float:
    """ME-weighted mean of `ret_excess` over rows in `group`."""
    g = group.dropna(subset=["me_lag1", "ret_excess"])
    w = g["me_lag1"].to_numpy(dtype="float64")
    r = g["ret_excess"].to_numpy(dtype="float64")
    total = w.sum()
    if total <= 0 or len(w) == 0:
        return np.nan
    return float((w * r).sum() / total)


def _factor_from_buckets(returns: pd.DataFrame, high_label: str, low_label: str) -> pd.Series:
    """Compute factor = ½(SH + BH) − ½(SL + BL) given a per-(date, S/B, char) table."""
    wide = returns.pivot_table(index="date", columns=["size_bucket", "char_bucket"], values="ret")
    rH = 0.5 * (wide.get(("S", high_label), np.nan) + wide.get(("B", high_label), np.nan))
    rL = 0.5 * (wide.get(("S", low_label), np.nan) + wide.get(("B", low_label), np.nan))
    return (rH - rL).rename("factor")


def _smb_from_beme_buckets(returns: pd.DataFrame) -> pd.Series:
    """SMB = ⅓(SL+SM+SH) − ⅓(BL+BM+BH) over BEME tertiles (Doc 3 single-sort variant)."""
    wide = returns.pivot_table(index="date", columns=["size_bucket", "char_bucket"], values="ret")
    s_avg = sum(wide.get(("S", lbl), np.nan) for lbl in ["L", "M", "H"]) / 3.0
    b_avg = sum(wide.get(("B", lbl), np.nan) for lbl in ["L", "M", "H"]) / 3.0
    return (s_avg - b_avg).rename("smb")


def _build_annual_axis(
    panel: pd.DataFrame,
    *,
    char_col: str,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> pd.DataFrame:
    """Compute per-(date, size, char) VW returns for an annually-rebalanced axis.

    Buckets are formed at each June-end; holdings persist for the next 12 months.
    """
    years = sorted(panel["date"].dt.year.unique())
    rebalance_dates = _june_endpoints(years)
    all_returns: list[pd.DataFrame] = []

    for june in rebalance_dates:
        snap = panel[panel["date"] == june]
        bucket_map = _assign_buckets(
            snap,
            char_col,
            low_label=low_label,
            mid_label=mid_label,
            high_label=high_label,
        )
        if bucket_map.empty:
            continue
        # Holding period: July(june.year) through June(june.year + 1)
        hold_start = june + pd.offsets.MonthEnd(1)
        hold_end = june + pd.DateOffset(years=1)
        holding = panel[(panel["date"] >= hold_start) & (panel["date"] <= hold_end)]
        holding = holding.merge(bucket_map, on="permno", how="inner")
        if holding.empty:
            continue
        agg = (
            holding.groupby(["date", "size_bucket", "char_bucket"], observed=True)
            .apply(_vw_return, include_groups=False)
            .reset_index(name="ret")
        )
        all_returns.append(agg)

    if not all_returns:
        return pd.DataFrame(columns=["date", "size_bucket", "char_bucket", "ret"])
    return pd.concat(all_returns, ignore_index=True)


def _build_monthly_axis(
    panel: pd.DataFrame,
    *,
    char_col: str,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> pd.DataFrame:
    """Compute per-(date, size, char) VW returns for a monthly-rebalanced axis (UMD)."""
    months = sorted(panel["date"].unique())
    all_returns: list[pd.DataFrame] = []

    for t in months:
        formation = pd.Timestamp(t) - pd.offsets.MonthEnd(1)
        snap_form = panel[panel["date"] == formation]
        bucket_map = _assign_buckets(
            snap_form,
            char_col,
            low_label=low_label,
            mid_label=mid_label,
            high_label=high_label,
        )
        if bucket_map.empty:
            continue
        snap_hold = panel[panel["date"] == t].merge(bucket_map, on="permno", how="inner")
        if snap_hold.empty:
            continue
        agg = (
            snap_hold.groupby(["size_bucket", "char_bucket"], observed=True)
            .apply(_vw_return, include_groups=False)
            .reset_index(name="ret")
            .assign(date=pd.Timestamp(t))
        )
        all_returns.append(agg)

    if not all_returns:
        return pd.DataFrame(columns=["date", "size_bucket", "char_bucket", "ret"])
    return pd.concat(all_returns, ignore_index=True)[["date", "size_bucket", "char_bucket", "ret"]]


def _build_mkt(panel: pd.DataFrame) -> pd.Series:
    """VW excess return of the universe, per month."""
    df = panel.dropna(subset=["me_lag1", "ret_excess"])
    result = (
        df.groupby("date")
        .apply(
            lambda g: (g["me_lag1"] * g["ret_excess"]).sum() / g["me_lag1"].sum(),
            include_groups=False,
        )
        .rename("mkt_ours")
    )
    return result


def build_ff_factors(
    panel: pd.DataFrame,
    *,
    sample_start: str = "2017-01-01",
    sample_end: str = "2024-12-31",
) -> pd.DataFrame:
    """Construct FF5 + UMD factors from our rank-normalized Stage 00 panel.

    The characteristic columns used for sorting are the rank-normalized
    versions in the panel (BEME, OP, Investment, r12_2). Because
    rank-normalization is monotonic in the underlying characteristic
    (per-month cross-section), NYSE 30 / 70 percentile bucket assignments
    on rank-z values match those on the raw characteristic.

    Returns
    -------
    DataFrame with columns:
        date, mkt_ours, smb_ours, hml_ours, rmw_ours, cma_ours, umd_ours
    """
    _validate_panel(panel)
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    # Lag ME by one month within permno for VW weighting.
    panel["me_lag1"] = panel.groupby("permno")["me"].shift(1)

    # Pad the window so that June-end rebalances and monthly UMD formation
    # months have data: pull from 1 year before sample_start through sample_end.
    pad_start = pd.Timestamp(sample_start) - pd.DateOffset(months=18)
    work = panel[(panel["date"] >= pad_start) & (panel["date"] <= sample_end)].copy()
    log.info(
        "ff: rows in working window %s..%s = %d",
        pad_start.date(),
        sample_end,
        len(work),
    )

    # Axis: HML (annual rebalance, BEME)
    hml_buckets = _build_annual_axis(
        work, char_col="BEME", low_label="L", mid_label="M", high_label="H"
    )
    hml = _factor_from_buckets(hml_buckets, high_label="H", low_label="L").rename("hml_ours")
    smb = _smb_from_beme_buckets(hml_buckets).rename("smb_ours")

    # Axis: RMW (annual rebalance, OP)
    rmw_buckets = _build_annual_axis(
        work, char_col="OP", low_label="W", mid_label="M", high_label="R"
    )
    rmw = _factor_from_buckets(rmw_buckets, high_label="R", low_label="W").rename("rmw_ours")

    # Axis: CMA (annual rebalance, Investment) -- reversed sign (C - A; C is low investment)
    cma_buckets = _build_annual_axis(
        work, char_col="Investment", low_label="C", mid_label="M", high_label="A"
    )
    cma = _factor_from_buckets(cma_buckets, high_label="C", low_label="A").rename("cma_ours")

    # Axis: UMD (monthly rebalance, r12_2)
    umd_buckets = _build_monthly_axis(
        work, char_col="r12_2", low_label="D", mid_label="M", high_label="U"
    )
    umd = _factor_from_buckets(umd_buckets, high_label="U", low_label="D").rename("umd_ours")

    # Market: VW of ret_excess
    mkt = _build_mkt(work).rename("mkt_ours")

    out = pd.concat([mkt, smb, hml, rmw, cma, umd], axis=1).reset_index()
    out = out[(out["date"] >= sample_start) & (out["date"] <= sample_end)].reset_index(drop=True)
    log.info("ff: factor table built; %d months, %d factors", len(out), out.shape[1] - 1)
    return out
