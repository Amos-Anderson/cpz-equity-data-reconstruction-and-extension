"""Annual accounting characteristic construction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_reconstruction.config import Stage00Config

log = logging.getLogger(__name__)


def _to_float(df: pd.DataFrame, skip: set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def compute_book_equity(comp: pd.DataFrame) -> pd.Series:
    """Davis-Fama-French book equity hierarchy."""
    ps = comp["pstkrv"].where(
        comp["pstkrv"].notna(),
        comp["pstkl"].where(comp["pstkl"].notna(), comp["pstk"].fillna(0.0)),
    )
    se = comp["seq"].where(
        comp["seq"].notna(),
        (comp["ceq"] + comp["pstk"].fillna(0.0)).where(comp["ceq"].notna(), comp["at"] - comp["lt"]),
    )
    be = se + comp["txditc"].fillna(0.0) - ps.fillna(0.0)
    return be.where(be > 0)


def compute_accounting_characteristics(comp: pd.DataFrame) -> pd.DataFrame:
    """Compute annual accounting characteristics on Compustat records."""
    comp = comp.copy()
    comp["datadate"] = pd.to_datetime(comp["datadate"])
    comp["permno"] = pd.to_numeric(comp["permno"], errors="coerce")
    comp = comp.dropna(subset=["permno"]).copy()
    comp["permno"] = comp["permno"].astype("int64")
    comp = _to_float(comp, {"gvkey", "datadate", "permno", "linktype", "linkprim", "sich"})
    comp = comp[comp["linkprim"].isin(["P", "C"])].copy()
    comp = comp.sort_values(["permno", "datadate"]).reset_index(drop=True)

    comp["ps"] = comp["pstkrv"].where(
        comp["pstkrv"].notna(),
        comp["pstkl"].where(comp["pstkl"].notna(), comp["pstk"].fillna(0.0)),
    )
    comp["se"] = comp["seq"].where(
        comp["seq"].notna(),
        (comp["ceq"] + comp["pstk"].fillna(0.0)).where(comp["ceq"].notna(), comp["at"] - comp["lt"]),
    )
    comp["be"] = compute_book_equity(comp)
    comp["avail_date"] = comp["datadate"] + pd.DateOffset(months=6) + pd.offsets.MonthEnd(0)

    group = comp.groupby("permno", group_keys=False)
    comp["be_lag1"] = group["be"].shift(1)
    comp["at_lag1"] = group["at"].shift(1)
    comp["at_lag2"] = group["at"].shift(2)
    comp["act_lag1"] = group["act"].shift(1)
    comp["lct_lag1"] = group["lct"].shift(1)
    comp["che_lag1"] = group["che"].shift(1)
    comp["dlc_lag1"] = group["dlc"].shift(1)

    comp["BE_raw"] = comp["be"]
    comp["IB_raw"] = comp["ib"]
    comp["CF_raw"] = comp["ib"].fillna(0.0) + comp["dp"].fillna(0.0)
    comp["DV_raw"] = comp["dv"]
    comp["SALE_raw"] = comp["sale"]
    comp["AT_raw"] = comp["at"]
    comp["DEBT_raw"] = comp["dltt"].fillna(0.0) + comp["dlc"].fillna(0.0)
    comp["AT_for_Q"] = comp["at"]
    comp["BE_for_Q"] = comp["be"]

    comp["PROF"] = ((comp["sale"].fillna(0.0) - comp["cogs"].fillna(0.0)) / comp["be"]).where(comp["be"] > 0)
    comp["ROE"] = (comp["ib"] / comp["be_lag1"]).where(comp["be_lag1"] > 0)
    comp["ROA"] = (comp["ib"] / comp["at_lag1"]).where(comp["at_lag1"] > 0)
    revt = comp["revt"].fillna(comp["sale"].fillna(0.0))
    comp["OP"] = ((revt - comp["cogs"].fillna(0.0) - comp["xsga"].fillna(0.0) - comp["xint"].fillna(0.0)) / comp["be"]).where(comp["be"] > 0)
    comp["PM"] = (comp["ib"] / comp["sale"]).where(comp["sale"].abs() > 0.01)
    comp["PCM"] = ((comp["sale"] - comp["cogs"].fillna(0.0)) / comp["sale"]).where(comp["sale"].abs() > 0.01)
    net_assets = comp["at"] - comp["che"].fillna(0.0)
    comp["RNA"] = (comp["ib"] / net_assets).where(net_assets.abs() > 0.01)

    comp["Investment"] = (comp["at"] / comp["at_lag1"] - 1.0).where(comp["at_lag1"] > 0)
    op_assets = comp["at"] - comp["che"].fillna(0.0) - comp["intan"].fillna(0.0)
    op_liab = comp["at"] - comp["dltt"].fillna(0.0) - comp["dlc"].fillna(0.0) - comp["pstk"].fillna(0.0) - comp["se"]
    comp["NOA"] = ((op_assets - op_liab) / comp["at_lag1"]).where(comp["at_lag1"] > 0)

    ppent_chg = comp["ppent"] - group["ppent"].shift(1)
    invt_chg = comp["invt"] - group["invt"].shift(1)
    comp["DPI2A"] = ((ppent_chg.fillna(0.0) + invt_chg.fillna(0.0)) / comp["at_lag1"]).where(comp["at_lag1"] > 0)

    d_act = comp["act"] - comp["act_lag1"]
    d_lct = comp["lct"] - comp["lct_lag1"]
    d_che = comp["che"] - comp["che_lag1"]
    d_dlc = comp["dlc"].fillna(0.0) - comp["dlc_lag1"].fillna(0.0)
    comp["OA"] = (((d_act.fillna(0.0) - d_che.fillna(0.0)) - (d_lct.fillna(0.0) - d_dlc.fillna(0.0)) - comp["dp"].fillna(0.0)) / comp["at_lag1"]).where(comp["at_lag1"] > 0)

    noa_curr = comp["NOA"] * comp["at_lag1"]
    noa_lag = group["NOA"].shift(1) * comp["at_lag2"]
    comp["AC"] = ((noa_curr - noa_lag) / comp["at_lag1"]).where(
        (comp["at_lag1"] > 0) & noa_curr.notna() & noa_lag.notna()
    )

    comp["C"] = (comp["che"].fillna(0.0) / comp["at"]).where(comp["at"] > 0)
    comp["AT"] = np.log(comp["at"].clip(lower=1e-6))
    comp["ATO"] = (comp["sale"] / comp["at_lag1"]).where(comp["at_lag1"] > 0)
    comp["CTO"] = (comp["sale"] / comp["ppent"]).where(comp["ppent"].fillna(0.0) > 0.01)
    comp["D2A"] = ((comp["dltt"].fillna(0.0) + comp["dlc"].fillna(0.0)) / comp["at"]).where(comp["at"] > 0)
    comp["FC2Y"] = ((comp["cogs"].fillna(0.0) + comp["xsga"].fillna(0.0)) / comp["at"]).where(comp["at"] > 0)
    comp["OL"] = comp["FC2Y"]
    comp["SGA2S"] = (comp["xsga"].fillna(0.0) / comp["sale"]).where(comp["sale"].abs() > 0.01)

    output_cols = [
        "permno",
        "datadate",
        "avail_date",
        "fyear",
        "BE_raw",
        "IB_raw",
        "CF_raw",
        "DV_raw",
        "SALE_raw",
        "AT_raw",
        "DEBT_raw",
        "AT_for_Q",
        "BE_for_Q",
        "PROF",
        "ROE",
        "ROA",
        "OP",
        "PM",
        "PCM",
        "RNA",
        "Investment",
        "NOA",
        "DPI2A",
        "OA",
        "AC",
        "C",
        "AT",
        "ATO",
        "CTO",
        "D2A",
        "FC2Y",
        "OL",
        "SGA2S",
    ]
    return comp[output_cols].sort_values(["permno", "avail_date"]).reset_index(drop=True)


def build_accounting_file(config: Stage00Config) -> pd.DataFrame:
    """Read raw Compustat, compute accounting characteristics, and save."""
    log.info("accounting: reading raw Compustat funda from %s", config.raw_dir)
    comp = pd.read_parquet(config.raw_dir / "compustat_annual_raw.parquet")
    log.info("accounting: %d raw Compustat rows", len(comp))
    comp_chars = compute_accounting_characteristics(comp)
    n_ac_defined = int(comp_chars["AC"].notna().sum())
    out_path = config.raw_dir / "accounting_chars.parquet"
    comp_chars.to_parquet(out_path, index=False)
    log.info(
        "accounting: %d firm-years; AC defined: %d (%.1f%%); -> %s",
        len(comp_chars),
        n_ac_defined,
        100.0 * n_ac_defined / max(len(comp_chars), 1),
        out_path,
    )
    return comp_chars

