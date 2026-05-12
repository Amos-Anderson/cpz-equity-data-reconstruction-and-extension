"""Ken French published factor returns: WRDS pull + local cache loader.

Ken French publishes the FF5 (MKTRF, SMB, HML, RMW, CMA), the momentum
factor (UMD), and the risk-free rate (RF) on his data library. WRDS
mirrors these in `ff.factors_monthly`. This module:

- Pulls the full FF5 + UMD + RF set into a single parquet via WRDS
  (idempotent; safe to call multiple times). The pull is *separable*
  from the broader Stage 00 raw download so the user does not need to
  refresh CRSP/Compustat to refresh the FF factors.
- Loads the cached parquet on subsequent runs.

The existing `ff_factors_monthly_full.parquet` (date, rf, mktrf, smb,
hml only) is left untouched. The extended pull writes a separate file,
`ff_factors_monthly_extended.parquet`, to avoid breaking other
consumers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_reconstruction.config import Stage00Config

log = logging.getLogger(__name__)


KF_EXTENDED_FILENAME = "ff_factors_monthly_extended.parquet"
KF_COLUMNS = ["date", "rf", "mktrf", "smb", "hml", "rmw", "cma", "umd"]


def _table_columns(conn, schema: str, table: str) -> set[str]:
    """Return the column names of `schema.table` on WRDS, lowercase."""
    df = conn.raw_sql(
        f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        """
    )
    if df is None or len(df) == 0:
        return set()
    return {str(c).lower() for c in df["column_name"]}


def _safe_pull(conn, sql: str) -> pd.DataFrame | None:
    """Try a WRDS pull; return None on any error (caller decides what to do)."""
    try:
        return conn.raw_sql(sql, date_cols=["date"])
    except Exception as exc:  # noqa: BLE001
        log.debug("kf: pull failed for SQL `%s...`: %s", sql[:120].replace("\n", " "), exc)
        return None


def pull_kf_extended(conn, config: Stage00Config) -> pd.DataFrame:
    """Pull MKTRF, SMB, HML, RMW, CMA, UMD, RF — robust across WRDS table layouts.

    The standard WRDS Fama-French library lays things out as:

    - ``ff.factors_monthly``       — FF3 (mktrf, smb, hml) + rf; sometimes umd.
    - ``ff.fivefactors_monthly``   — FF5 (mktrf, smb, hml, rmw, cma) + rf.
    - ``ff.factors_monthly`` umd   — present on some WRDS subscriptions; absent
                                     on others. We try here and fall back to
                                     a separate momentum table if needed.

    This function:

    1. Pulls FF3 + RF from ``ff.factors_monthly`` (known to work).
    2. Pulls RMW + CMA from ``ff.fivefactors_monthly``.
    3. Looks for UMD in ``ff.factors_monthly`` first, then in
       ``ff.liq_ps`` / ``ff.factors_monthly_mom`` / ``ff.momentum_monthly``.
    4. Merges on date and saves to ``ff_factors_monthly_extended.parquet``.

    Missing factors are logged at WARNING; the parquet is still written with
    whatever was found so Arm 2 can compare the available factors.
    """
    config.ensure_dirs()
    start, end = config.crsp_monthly_start, config.crsp_monthly_end
    log.info("kf: pulling FF5 + UMD + RF (%s..%s)", start, end)

    # 1. FF3 + RF from ff.factors_monthly
    base = conn.raw_sql(
        f"""
        SELECT date, rf, mktrf, smb, hml
        FROM ff.factors_monthly
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date
        """,
        date_cols=["date"],
    )
    log.info("kf: ff.factors_monthly (FF3+RF) rows=%d", len(base))

    # 2. RMW + CMA from ff.fivefactors_monthly
    ff5 = _safe_pull(
        conn,
        f"""
        SELECT date, rmw, cma
        FROM ff.fivefactors_monthly
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date
        """,
    )
    if ff5 is None:
        log.warning("kf: ff.fivefactors_monthly unavailable; RMW and CMA will be missing")
        ff5 = pd.DataFrame(columns=["date", "rmw", "cma"])
    else:
        log.info("kf: ff.fivefactors_monthly (RMW+CMA) rows=%d", len(ff5))

    # 3. UMD: try ff.factors_monthly first, then known alternative tables.
    umd: pd.DataFrame | None = None
    candidates = [
        ("ff.factors_monthly", "umd"),
        ("ff.factors_monthly_umd", "umd"),
        ("ff.factors_mom_monthly", "umd"),
        ("ff.momentum_monthly", "umd"),
        ("ff.factors_monthly", "mom"),  # some installs alias to mom
    ]
    for schema_table, col in candidates:
        try:
            schema, table = schema_table.split(".")
            cols = _table_columns(conn, schema, table)
        except Exception:  # noqa: BLE001
            continue
        if col in cols:
            df_try = _safe_pull(
                conn,
                f"""
                SELECT date, {col} AS umd
                FROM {schema_table}
                WHERE date BETWEEN '{start}' AND '{end}'
                ORDER BY date
                """,
            )
            if df_try is not None and len(df_try) > 0:
                umd = df_try
                log.info("kf: pulled UMD from %s.%s (%d rows)", schema_table, col, len(umd))
                break
    if umd is None:
        log.warning("kf: UMD column not found in any candidate table; UMD comparison will be unavailable")

    # 4. Merge
    df = base
    df = df.merge(ff5, on="date", how="left")
    if umd is not None:
        df = df.merge(umd, on="date", how="left")
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)

    # Ensure every expected column exists (NaN if a factor was unavailable)
    for c in ["rf", "mktrf", "smb", "hml", "rmw", "cma", "umd"]:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[KF_COLUMNS]

    out = config.raw_dir / KF_EXTENDED_FILENAME
    df.to_parquet(out, index=False)
    log.info("kf: saved %d months × %d cols -> %s", len(df), df.shape[1], out)
    log.info(
        "kf: non-null counts %s",
        {c: int(df[c].notna().sum()) for c in ["mktrf", "smb", "hml", "rmw", "cma", "umd"]},
    )
    return df


def pull_kf_extended_standalone(config: Stage00Config) -> pd.DataFrame:
    """Convenience: open a WRDS connection just to pull the KF extended file."""
    import wrds

    log.info("kf: opening WRDS connection for KF extended pull")
    conn = wrds.Connection()
    try:
        return pull_kf_extended(conn, config)
    finally:
        conn.close()


def load_kf_factors(raw_dir: Path) -> pd.DataFrame:
    """Load the cached Ken French factor parquet.

    Raises
    ------
    FileNotFoundError
        If the extended parquet is missing. The error message points the
        user to ``python -m data_reconstruction.pipeline --pull-kf``.
    ValueError
        If the parquet is missing one of the expected columns.
    """
    p = Path(raw_dir) / KF_EXTENDED_FILENAME
    if not p.exists():
        raise FileNotFoundError(
            f"Ken French extended factor parquet not found at {p}. "
            "Run `python -m data_reconstruction.pipeline --pull-kf` once "
            "to fetch it (requires WRDS access)."
        )
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    missing = [c for c in KF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"KF extended parquet at {p} missing columns: {missing}. "
            "Run `python -m data_reconstruction.pipeline --pull-kf` to refresh."
        )
    return df[KF_COLUMNS].copy()
