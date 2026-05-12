"""WRDS raw data downloads.

The SQL is intentionally close to the previous working notebook.  These
functions only pull and save raw parquet files; construction happens in
later modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_reconstruction.config import Stage00Config

log = logging.getLogger(__name__)


def _financials_filter(exclude_financials: bool, alias: str = "n") -> str:
    if not exclude_financials:
        return ""
    return f"AND ({alias}.siccd < 6000 OR {alias}.siccd > 6999)"


def pull_crsp_monthly(conn, config: Stage00Config) -> pd.DataFrame:
    """Pull CRSP monthly stock data and delisting returns."""
    config.ensure_dirs()
    fin_filter = _financials_filter(config.exclude_financials, "n")
    msf = conn.raw_sql(
        f"""
        SELECT
            m.permno,
            m.date,
            m.ret,
            m.retx,
            ABS(m.prc) AS prc,
            m.shrout,
            m.vol,
            m.cfacpr,
            m.cfacshr,
            n.shrcd,
            n.exchcd,
            n.siccd
        FROM crsp.msf m
        INNER JOIN crsp.msenames n
            ON  m.permno = n.permno
            AND m.date BETWEEN n.namedt AND n.nameendt
        WHERE m.date BETWEEN '{config.crsp_monthly_start}' AND '{config.crsp_monthly_end}'
          AND n.shrcd IN (10, 11)
          AND n.exchcd IN (1, 2, 3)
          {fin_filter}
        ORDER BY m.permno, m.date
        """,
        date_cols=["date"],
    )
    msf.to_parquet(config.raw_dir / "crsp_msf_raw.parquet", index=False)

    delist = conn.raw_sql(
        f"""
        SELECT permno, dlstdt AS date, dlret, dlstcd
        FROM crsp.msedelist
        WHERE dlstdt BETWEEN '{config.crsp_monthly_start}' AND '{config.crsp_monthly_end}'
        ORDER BY permno, dlstdt
        """,
        date_cols=["date"],
    )
    delist.to_parquet(config.raw_dir / "crsp_delist_raw.parquet", index=False)
    return msf


def pull_compustat(conn, config: Stage00Config) -> pd.DataFrame:
    """Pull Compustat annual fundamentals with CCM links."""
    config.ensure_dirs()
    comp = conn.raw_sql(
        f"""
        SELECT
            c.gvkey,
            c.datadate,
            c.fyear,
            c.fyr,
            c.sich,
            l.lpermno AS permno,
            l.linktype,
            l.linkprim,
            c.at, c.lt,
            c.seq, c.ceq, c.pstk,
            c.pstkrv, c.pstkl, c.txditc,
            c.sale, c.revt, c.cogs, c.xsga,
            c.xint, c.ib, c.ni, c.oiadp, c.gp,
            c.oancf, c.capx, c.dp, c.dv,
            c.act, c.che, c.rect, c.invt,
            c.ppent, c.intan, c.ao,
            c.lct, c.dlc, c.dltt,
            c.ap, c.txp, c.lo,
            c.csho, c.ajex, c.re, c.txdb,
            c.xrd, c.emp
        FROM comp.funda c
        INNER JOIN crsp.ccmxpf_linktable l
            ON  c.gvkey = l.gvkey
            AND c.datadate BETWEEN l.linkdt AND COALESCE(l.linkenddt, '2024-12-31')
            AND l.linktype IN ('LC', 'LU')
            AND l.linkprim IN ('P', 'C')
        WHERE c.datadate BETWEEN '{config.compustat_start}' AND '{config.compustat_end}'
          AND c.indfmt = 'INDL'
          AND c.datafmt = 'STD'
          AND c.popsrc = 'D'
          AND c.consol = 'C'
        ORDER BY l.lpermno, c.datadate
        """,
        date_cols=["datadate"],
    )
    comp.to_parquet(config.raw_dir / "compustat_annual_raw.parquet", index=False)
    return comp


def pull_factor_and_daily_data(conn, config: Stage00Config) -> dict[str, pd.DataFrame]:
    """Pull FF factors, CRSP market index, and CRSP daily data."""
    config.ensure_dirs()
    ff_monthly = conn.raw_sql(
        """
        SELECT date, rf, mktrf, smb, hml
        FROM ff.factors_monthly
        WHERE date BETWEEN '1926-01-01' AND '2024-12-31'
        ORDER BY date
        """,
        date_cols=["date"],
    )
    ff_monthly["date"] = pd.to_datetime(ff_monthly["date"]) + pd.offsets.MonthEnd(0)
    ff_monthly.to_parquet(config.raw_dir / "ff_factors_monthly_full.parquet", index=False)

    ff_daily = conn.raw_sql(
        f"""
        SELECT date, mktrf, rf, smb, hml
        FROM ff.factors_daily
        WHERE date BETWEEN '{config.crsp_daily_start}' AND '{config.crsp_daily_end}'
        ORDER BY date
        """,
        date_cols=["date"],
    )
    ff_daily["date"] = pd.to_datetime(ff_daily["date"])
    ff_daily.to_parquet(config.raw_dir / "ff_factors_daily.parquet", index=False)

    market = conn.raw_sql(
        f"""
        SELECT date, vwretd AS mkt_ret, ewretd AS ew_ret
        FROM crsp.msi
        WHERE date BETWEEN '{config.crsp_monthly_start}' AND '{config.crsp_monthly_end}'
        ORDER BY date
        """,
        date_cols=["date"],
    )
    market["date"] = pd.to_datetime(market["date"]) + pd.offsets.MonthEnd(0)
    market.to_parquet(config.raw_dir / "crsp_market_index.parquet", index=False)

    fin_filter = _financials_filter(config.exclude_financials, "n")
    dsf = conn.raw_sql(
        f"""
        SELECT
            d.permno,
            d.date,
            d.ret,
            ABS(d.prc) AS prc,
            d.vol,
            d.shrout,
            d.ask,
            d.bid
        FROM crsp.dsf d
        INNER JOIN crsp.msenames n
            ON  d.permno = n.permno
            AND d.date BETWEEN n.namedt AND n.nameendt
        WHERE d.date BETWEEN '{config.crsp_daily_start}' AND '{config.crsp_daily_end}'
          AND n.shrcd IN (10, 11)
          AND n.exchcd IN (1, 2, 3)
          {fin_filter}
        ORDER BY d.permno, d.date
        """,
        date_cols=["date"],
    )
    dsf.to_parquet(config.raw_dir / "crsp_dsf_raw.parquet", index=False)
    return {"ff_monthly": ff_monthly, "ff_daily": ff_daily, "market": market, "dsf": dsf}


def pull_all_raw_data(config: Stage00Config, conn=None) -> None:
    """Pull all raw WRDS files used by the pipeline."""
    close_conn = False
    if conn is None:
        import wrds

        log.info("download: opening WRDS connection")
        conn = wrds.Connection()
        close_conn = True
    try:
        log.info("download: pulling CRSP MSF + delist (%s..%s)", config.crsp_monthly_start, config.crsp_monthly_end)
        pull_crsp_monthly(conn, config)
        log.info("download: pulling Compustat funda (%s..%s)", config.compustat_start, config.compustat_end)
        pull_compustat(conn, config)
        log.info("download: pulling FF monthly/daily + CRSP daily")
        pull_factor_and_daily_data(conn, config)
        log.info("download: all raw pulls complete -> %s", config.raw_dir)
    finally:
        if close_conn:
            conn.close()

