"""Lightweight tests for the simple Stage 00 modular rewrite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_reconstruction.accounting import compute_book_equity
from data_reconstruction.assemble import apply_completeness_filter, rank_normalize
from data_reconstruction.config import Stage00Config
from data_reconstruction.constants import ALL_46
from data_reconstruction.crsp import build_clean_monthly
from data_reconstruction.risk import compute_roll_spread
from data_reconstruction.validation import (
    annual_breadth_table,
    annual_returns_table,
    audit_coverage_stability,
    audit_pit_bounds,
    audit_rank_normalization,
    audit_schema,
    build_ff_factors,
    char_stats_table,
    evaluate_acceptance,
    per_factor_diagnostics,
)


def test_book_equity_hierarchy_and_positive_filter() -> None:
    comp = pd.DataFrame(
        {
            "seq": [100.0, np.nan, np.nan, 5.0],
            "ceq": [np.nan, 80.0, np.nan, np.nan],
            "pstk": [10.0, 2.0, 0.0, 10.0],
            "at": [200.0, 150.0, 120.0, 50.0],
            "lt": [70.0, 60.0, 40.0, 10.0],
            "pstkrv": [8.0, np.nan, np.nan, np.nan],
            "pstkl": [np.nan, 3.0, np.nan, np.nan],
            "txditc": [5.0, 0.0, 2.0, 0.0],
        }
    )
    be = compute_book_equity(comp)
    assert be.iloc[0] == 97.0
    assert be.iloc[1] == 79.0
    assert be.iloc[2] == 82.0
    assert pd.isna(be.iloc[3])


def test_completeness_filter_requires_all_characteristics() -> None:
    panel = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [1.0, np.nan, 3.0]})
    filtered = apply_completeness_filter(panel, ["A", "B"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["A"] == 1.0


def test_rank_normalize_open_interval() -> None:
    values = pd.Series([10.0, 20.0, 30.0])
    out = rank_normalize(values)
    assert out.min() > -0.5
    assert out.max() < 0.5
    assert out.tolist() == [-0.25, 0.0, 0.25]


def test_clean_monthly_delisting_compounds_returns(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = Stage00Config(raw_dir=raw_dir, output_dir=tmp_path / "out", foundation_dir=tmp_path / "foundation")

    pd.DataFrame(
        {
            "permno": [1],
            "date": [pd.Timestamp("2020-01-31")],
            "ret": [0.10],
            "retx": [0.10],
            "prc": [20.0],
            "shrout": [1000.0],
            "vol": [10000.0],
            "cfacpr": [1.0],
            "cfacshr": [1.0],
            "shrcd": [10],
            "exchcd": [3],
            "siccd": [1000],
        }
    ).to_parquet(raw_dir / "crsp_msf_raw.parquet", index=False)
    pd.DataFrame(
        {
            "permno": [1],
            "date": [pd.Timestamp("2020-01-31")],
            "dlret": [np.nan],
            "dlstcd": [520],
        }
    ).to_parquet(raw_dir / "crsp_delist_raw.parquet", index=False)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")],
            "rf": [0.001],
            "mktrf": [0.02],
        }
    ).to_parquet(raw_dir / "ff_factors_monthly_full.parquet", index=False)

    clean = build_clean_monthly(config)
    expected_ret_adj = (1.10 * 0.45) - 1.0
    assert clean.loc[0, "ret_adj"] == pytest.approx(expected_ret_adj)
    assert clean.loc[0, "ret_excess"] == pytest.approx(expected_ret_adj - 0.001)
    assert clean.loc[0, "me"] == 20.0


def test_compute_roll_spread_names_output_column() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="B")
    daily = pd.DataFrame(
        {
            "permno": [1] * len(dates),
            "date": dates,
            "ret": [0.01, -0.01] * 6,
        }
    )
    spread = compute_roll_spread(daily)
    assert list(spread.columns) == ["permno", "date", "Spread"]
    assert len(spread) == 1
    assert spread.loc[0, "Spread"] > 0


# -----------------------------------------------------------------------------
# Arm 3 audit tests
# -----------------------------------------------------------------------------


def _make_audit_panel(rows: int = 6) -> pd.DataFrame:
    """Synthetic panel that satisfies all Arm 3 audits."""
    datadate = pd.Timestamp("2019-12-31")
    avail = (datadate + pd.DateOffset(months=6)) + pd.offsets.MonthEnd(0)
    dates = pd.date_range(avail, periods=rows, freq="ME")
    permnos = [1, 2] * (rows // 2 + 1)
    n = rows
    panel = pd.DataFrame(
        {
            "permno": np.tile([1, 2], n // 2 + 1)[:n],
            "date": np.tile(dates, 2)[:n],
            "ret_excess": np.linspace(-0.05, 0.05, n),
            "me": np.linspace(1e6, 1e8, n),
            "datadate": [datadate] * n,
        }
    )
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    rng = np.random.default_rng(seed=0)
    for col in ALL_46:
        ranks = rng.integers(1, n + 1, size=n)
        panel[col] = ranks / (n + 1.0) - 0.5
    return panel


def test_audit_pit_bounds_pass() -> None:
    panel = _make_audit_panel()
    result = audit_pit_bounds(panel)
    assert result.passed
    assert result.n_violations == 0
    assert result.severity == "hard"


def test_audit_pit_bounds_detects_too_old() -> None:
    panel = _make_audit_panel()
    panel.loc[0, "datadate"] = pd.Timestamp("2015-01-31")
    result = audit_pit_bounds(panel)
    assert not result.passed
    assert result.n_violations == 1
    assert result.sample_violations[0]["date"] == panel.loc[0, "date"].strftime("%Y-%m-%d")


def test_audit_pit_bounds_detects_too_new() -> None:
    panel = _make_audit_panel()
    panel.loc[0, "datadate"] = panel.loc[0, "date"] + pd.DateOffset(years=1)
    result = audit_pit_bounds(panel)
    assert not result.passed
    assert result.n_violations >= 1


def test_audit_rank_normalization_pass() -> None:
    panel = _make_audit_panel()
    result = audit_rank_normalization(panel)
    assert result.passed
    assert result.n_violations == 0


def test_audit_rank_normalization_detects_out_of_range() -> None:
    panel = _make_audit_panel()
    panel.loc[0, "BEME"] = 0.5
    result = audit_rank_normalization(panel)
    assert not result.passed
    assert result.n_violations == 1


def test_audit_schema_detects_duplicates() -> None:
    panel = _make_audit_panel()
    dup_row = panel.iloc[[0]].copy()
    panel = pd.concat([panel, dup_row], ignore_index=True)
    result = audit_schema(panel)
    assert not result.passed
    assert "duplicate" in result.message


def test_audit_coverage_stability_flags_jumps() -> None:
    # Build a panel with a clean 1986 step-up (accepted) and a 2000 spike (hard).
    months = pd.date_range("1985-01-31", "2001-12-31", freq="ME")
    rows = []
    for d in months:
        if d.year < 1986:
            n_firms = 200
        elif d.year < 2000:
            n_firms = 400
        else:
            n_firms = 800
        for p in range(n_firms):
            rows.append({"permno": p + 1000 * d.year, "date": d})
    panel = pd.DataFrame(rows)
    result = audit_coverage_stability(panel)
    # The 1986 jump is on the known list; the 2000 doubling is not.
    assert not result.passed
    assert result.severity == "hard"


# -----------------------------------------------------------------------------
# Arm 1 distributional comparison tests
# -----------------------------------------------------------------------------


def _make_two_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic ours+cpz panels with known breadth and return differences."""
    months = pd.date_range("2000-01-31", "2001-12-31", freq="ME")
    ours_rows = []
    cpz_rows = []
    rng = np.random.default_rng(seed=42)
    for d in months:
        n_ours = 200
        n_cpz = 180
        for p in range(n_ours):
            row = {"permno": p + 1, "date": d, "ret_excess": rng.normal(0.01, 0.05)}
            for c in ALL_46:
                row[c] = rng.uniform(-0.49, 0.49)
            ours_rows.append(row)
        for _ in range(n_cpz):
            row = {"date": d, "ret": rng.normal(0.015, 0.05)}
            for c in ALL_46:
                row[c] = rng.uniform(-0.49, 0.49)
            cpz_rows.append(row)
    return pd.DataFrame(ours_rows), pd.DataFrame(cpz_rows)


def test_annual_breadth_table_shape_and_diff() -> None:
    ours, cpz = _make_two_panels()
    out = annual_breadth_table(ours, cpz)
    assert set(out.columns) == {"year", "ours_breadth", "cpz_breadth", "diff", "pct_diff"}
    assert len(out) == 2  # 2000, 2001
    # Ours has 200 firms / month, CPZ has 180
    assert (out["ours_breadth"] - 200).abs().max() < 1e-9
    assert (out["cpz_breadth"] - 180).abs().max() < 1e-9
    assert (out["diff"] - 20).abs().max() < 1e-9


def test_annual_returns_table_uses_correct_columns() -> None:
    ours, cpz = _make_two_panels()
    out = annual_returns_table(ours, cpz)
    assert set(out.columns) == {"year", "ours_ann_ret", "cpz_ann_ret", "diff"}
    assert len(out) == 2
    # ours mean ~ 0.01, cpz mean ~ 0.015; diff should be ~ -0.005
    assert out["ours_ann_ret"].mean() < out["cpz_ann_ret"].mean()


def test_char_stats_table_skips_missing_chars() -> None:
    ours, cpz = _make_two_panels()
    # Drop a char from cpz to verify the function skips it
    cpz_missing = cpz.drop(columns=["BEME"])
    out = char_stats_table(ours, cpz_missing, ALL_46)
    assert "BEME" not in out["characteristic"].values
    assert "E2P" in out["characteristic"].values
    assert len(out) == 45


# -----------------------------------------------------------------------------
# Arm 2: FF5 + UMD construction and comparison
# -----------------------------------------------------------------------------


def _make_ff_panel(n_firms: int = 200, months: int = 36, seed: int = 7) -> pd.DataFrame:
    """Synthetic monthly panel with the columns FF construction requires.

    Returns a panel with 200 firms over 36 months, exchcd diversified
    so a NYSE breakpoint can be computed.
    """
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range("2016-01-31", periods=months, freq="ME")
    rows = []
    for p in range(n_firms):
        exchcd = 1 if p < 80 else (2 if p < 130 else 3)  # 80 NYSE, 50 AMEX, 70 NASDAQ
        for d in dates:
            rows.append(
                {
                    "permno": p + 1,
                    "date": d,
                    "exchcd": exchcd,
                    "me": rng.uniform(1e6, 1e10),
                    "ret_excess": rng.normal(0.01, 0.05),
                    "BEME": rng.uniform(-0.49, 0.49),
                    "OP": rng.uniform(-0.49, 0.49),
                    "Investment": rng.uniform(-0.49, 0.49),
                    "r12_2": rng.uniform(-0.49, 0.49),
                }
            )
    return pd.DataFrame(rows)


def test_build_ff_factors_returns_expected_schema() -> None:
    panel = _make_ff_panel()
    ff = build_ff_factors(panel, sample_start="2017-01-31", sample_end="2018-12-31")
    expected = {"date", "mkt_ours", "smb_ours", "hml_ours", "rmw_ours", "cma_ours", "umd_ours"}
    assert expected.issubset(set(ff.columns))
    # All factor series have at least 12 non-null months in the test window
    for col in ["mkt_ours", "smb_ours", "hml_ours", "rmw_ours", "cma_ours", "umd_ours"]:
        assert ff[col].notna().sum() >= 12, f"{col} has too few non-null values"


def test_per_factor_diagnostics_shape_and_pearson_self() -> None:
    # Construct a joined table where ours == kf exactly; rho should be 1.0
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(seed=11)
    df = pd.DataFrame(
        {
            "date": dates,
            "mkt_ours": rng.normal(0.01, 0.04, 24),
            "smb_ours": rng.normal(0.0, 0.03, 24),
            "hml_ours": rng.normal(0.0, 0.03, 24),
            "rmw_ours": rng.normal(0.0, 0.02, 24),
            "cma_ours": rng.normal(0.0, 0.02, 24),
            "umd_ours": rng.normal(0.0, 0.04, 24),
        }
    )
    df["mktrf"] = df["mkt_ours"]
    df["smb"] = df["smb_ours"]
    df["hml"] = df["hml_ours"]
    df["rmw"] = df["rmw_ours"]
    df["cma"] = df["cma_ours"]
    df["umd"] = df["umd_ours"]
    diag = per_factor_diagnostics(df)
    assert len(diag) == 6
    assert (diag["pearson_rho"] > 0.999).all()
    assert (diag["mad"] < 1e-9).all()


def test_evaluate_acceptance_distinguishes_pass_and_fail() -> None:
    diag_pass = pd.DataFrame(
        {
            "factor": ["mkt", "smb", "hml", "rmw", "cma", "umd"],
            "pearson_rho": [0.99, 0.98, 0.97, 0.96, 0.95, 0.92],
            "mad": [0.005] * 6,
            "delta_ann": [0.01] * 6,
            "min_year_rho": [0.95, 0.94, 0.92, 0.91, 0.90, 0.88],
        }
    )
    assert evaluate_acceptance(diag_pass)["all_passed"] is True
    diag_fail = diag_pass.copy()
    diag_fail.loc[0, "pearson_rho"] = 0.50
    result = evaluate_acceptance(diag_fail)
    assert result["all_passed"] is False
    assert "mkt" in result["criteria"]["pearson_gt_0p85"]["failing"]
