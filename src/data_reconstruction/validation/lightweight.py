"""Lightweight CPZ breadth/return diagnostic.

This is a sanity check, not part of the Doc 3 acceptance bar. It compares
yearly firm counts and excess-return moments between the reconstructed
panel and the published CPZ reference panel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from data_reconstruction.config import Stage00Config

log = logging.getLogger(__name__)


def cpz_target_summary(cpz_path: Path) -> dict[str, object]:
    """Summarize breadth and returns from the original CPZ reference panel.

    Parameters
    ----------
    cpz_path
        Path to the CPZ-published characteristic panel (Parquet).

    Returns
    -------
    dict
        Mapping with `breadth` (year -> mean firm count) and the five
        return quantile / moment statistics.
    """
    cpz = pd.read_parquet(cpz_path)
    cpz["date"] = pd.to_datetime(cpz["date"]) + pd.offsets.MonthEnd(0)
    cpz["year"] = cpz["date"].dt.year
    year_group = cpz.groupby("year")
    breadth = year_group.size() / year_group["date"].nunique()
    ret = cpz["ret"].dropna()
    return {
        "breadth": {int(k): float(v) for k, v in breadth.items()},
        "ret_mean": float(ret.mean()),
        "ret_std": float(ret.std()),
        "ret_p10": float(ret.quantile(0.10)),
        "ret_p50": float(ret.quantile(0.50)),
        "ret_p90": float(ret.quantile(0.90)),
    }


def validate_breadth_and_returns(panel: pd.DataFrame, config: Stage00Config) -> dict[str, object]:
    """Compare broad Stage 00 output diagnostics to CPZ over 1967-2016."""
    cpz = cpz_target_summary(config.cpz_reference_path)
    sample = panel[(panel["date"] >= "1967-01-01") & (panel["date"] <= "2016-12-31")].copy()
    sample["year"] = sample["date"].dt.year
    year_group = sample.groupby("year")
    breadth = year_group.size() / year_group["date"].nunique()
    ret = sample["ret_excess"].dropna()
    out: dict[str, object] = {
        "ours_breadth": {int(k): float(v) for k, v in breadth.items()},
        "cpz_breadth": cpz["breadth"],
        "ours_ret_mean": float(ret.mean()),
        "ours_ret_std": float(ret.std()),
        "ours_ret_p10": float(ret.quantile(0.10)),
        "ours_ret_p50": float(ret.quantile(0.50)),
        "ours_ret_p90": float(ret.quantile(0.90)),
        "cpz_ret_mean": cpz["ret_mean"],
        "cpz_ret_std": cpz["ret_std"],
        "cpz_ret_p10": cpz["ret_p10"],
        "cpz_ret_p50": cpz["ret_p50"],
        "cpz_ret_p90": cpz["ret_p90"],
    }
    config.foundation_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.foundation_dir / "cpz_breadth_return_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("lightweight CPZ breadth/return summary → %s", out_path)
    return out
