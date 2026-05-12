"""Arm 3: internal consistency audits.

Five audits per `00_data_foundation_doc3_validation.md` §"Arm 3 — Internal
Consistency Audits":

- 3.1 PIT bounds — every row satisfies
  `avail_date <= date <= avail_date + tolerance_days`, where
  `avail_date = datadate + 6 months`. Equivalent to Doc 1's "no stale-by-more-than
  tolerance" rule. See note in `audit_pit_bounds` for the Doc 3 literal-formula
  caveat.
- 3.2 Look-ahead — handled by construction (rolling windows in `monthly.py`
  and `risk.py` use shifted endpoints). Not implemented here as a data audit;
  enforced via unit tests of the builders.
- 3.3 Coverage stability — monthly firm counts change smoothly. Known
  acceptable jumps: 1986 NASDAQ-Compustat integration. Anything else with
  |YoY change| > 30 percent is a hard-fail.
- 3.4 Rank normalization — every characteristic value lies strictly in the
  open interval (-1/2, +1/2).
- 3.5 Output schema — required columns present with correct dtypes; no
  duplicate (permno, date); dates monotonic within permno.

Each audit returns an `AuditResult`. `run_all_audits` collects them and
serializes to `arm3_summary.json`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_reconstruction.constants import ALL_46

log = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """One audit's pass/fail status, counts, and a sample of violations."""

    name: str
    passed: bool
    severity: str  # "hard" | "soft" | "info"
    n_checked: int
    n_violations: int
    message: str
    sample_violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_pit_bounds(
    panel: pd.DataFrame,
    *,
    tolerance_days: int = 380,
    lag_months: int = 6,
    sample_size: int = 5,
) -> AuditResult:
    """Verify every row satisfies the PIT bound enforced by `merge_asof`.

    The bound is `avail_date <= date <= avail_date + tolerance_days`, where
    `avail_date = datadate + lag_months` rounded up to month-end. This matches
    the actual `pd.merge_asof(..., direction="backward", tolerance=...)` join
    in `assemble.assemble_panel`.

    Note
    ----
    Doc 3 §"Audit 3.1" literally writes
    `datadate + 6m <= date < datadate + 380 days`. That upper bound is
    `datadate + 12.5m`, which leaves only ~6.5 months of usable life after
    the 6-month publication lag — narrower than what annual filings allow.
    Doc 1 §"Audit fix 1" describes the 380-day tolerance as "12.5 months
    *stale*", i.e. measured from `avail_date` not from `datadate`. The
    implementation (and this audit) follow Doc 1's wider window, which is
    what the `merge_asof` tolerance actually enforces.
    """
    cols = ["datadate", "date"]
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        return AuditResult(
            name="pit_bounds",
            passed=False,
            severity="hard",
            n_checked=0,
            n_violations=len(panel),
            message=f"missing required columns: {missing}",
        )

    sub = panel[cols].copy()
    sub["datadate"] = pd.to_datetime(sub["datadate"])
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.dropna(subset=cols)
    n_checked = int(len(sub))

    avail = (sub["datadate"] + pd.DateOffset(months=lag_months)) + pd.offsets.MonthEnd(0)
    upper = avail + pd.Timedelta(days=tolerance_days)
    below = sub["date"] < avail
    above = sub["date"] > upper
    violations = sub[below | above].copy()
    violations["avail_date_expected"] = avail[below | above]
    violations["upper_bound"] = upper[below | above]

    n_violations = int(len(violations))
    passed = n_violations == 0
    msg = (
        f"PIT bounds OK on {n_checked:,} rows"
        if passed
        else f"{n_violations:,} of {n_checked:,} rows violate avail_date <= date <= avail_date + {tolerance_days}d"
    )
    sample = (
        violations.head(sample_size)
        .assign(
            datadate=lambda d: d["datadate"].dt.strftime("%Y-%m-%d"),
            date=lambda d: d["date"].dt.strftime("%Y-%m-%d"),
            avail_date_expected=lambda d: d["avail_date_expected"].dt.strftime("%Y-%m-%d"),
            upper_bound=lambda d: d["upper_bound"].dt.strftime("%Y-%m-%d"),
        )
        .to_dict(orient="records")
    )
    return AuditResult(
        name="pit_bounds",
        passed=passed,
        severity="hard",
        n_checked=n_checked,
        n_violations=n_violations,
        message=msg,
        sample_violations=sample,
    )


def audit_rank_normalization(
    panel: pd.DataFrame,
    *,
    char_cols: list[str] | None = None,
    eps: float = 1e-12,
    sample_size: int = 5,
) -> AuditResult:
    """Verify every characteristic value lies strictly in (-1/2, +1/2).

    Per Doc 3 §"Audit 3.4", a value of exactly +/- 0.5 indicates a
    rank-normalization bug (the formula `R/(N+1) - 1/2` is strictly
    bounded away from the endpoints).
    """
    cols = char_cols or ALL_46
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        return AuditResult(
            name="rank_normalization",
            passed=False,
            severity="hard",
            n_checked=0,
            n_violations=len(panel) * len(missing),
            message=f"missing characteristic columns: {missing}",
        )

    arr = panel[cols].to_numpy(dtype="float64")
    finite = np.isfinite(arr)
    too_low = (arr <= -0.5 + eps) & finite
    too_high = (arr >= 0.5 - eps) & finite
    bad = too_low | too_high

    n_checked = int(finite.sum())
    n_violations = int(bad.sum())
    passed = n_violations == 0

    per_col = {
        cols[j]: int(bad[:, j].sum())
        for j in range(arr.shape[1])
        if bad[:, j].any()
    }
    sample = []
    if n_violations:
        rows_with_violations = np.where(bad.any(axis=1))[0][:sample_size]
        for r in rows_with_violations:
            row_record = {
                "row_index": int(r),
                "permno": int(panel.iloc[r]["permno"]) if "permno" in panel.columns else None,
                "date": (
                    panel.iloc[r]["date"].strftime("%Y-%m-%d")
                    if "date" in panel.columns and pd.notna(panel.iloc[r]["date"])
                    else None
                ),
                "violating_chars": {
                    cols[j]: float(arr[r, j])
                    for j in range(arr.shape[1])
                    if bad[r, j]
                },
            }
            sample.append(row_record)

    msg = (
        f"rank-norm OK on {n_checked:,} (row, char) cells"
        if passed
        else f"{n_violations:,} cells outside (-1/2, +1/2) across {len(per_col)} characteristics: {per_col}"
    )
    return AuditResult(
        name="rank_normalization",
        passed=passed,
        severity="hard",
        n_checked=n_checked,
        n_violations=n_violations,
        message=msg,
        sample_violations=sample,
    )


def audit_schema(
    panel: pd.DataFrame,
    *,
    required_id_cols: tuple[str, ...] = ("permno", "date", "ret_excess", "me"),
    char_cols: list[str] | None = None,
    sample_size: int = 5,
) -> AuditResult:
    """Schema integrity: required columns, no dup keys, monotonic dates."""
    cols = char_cols or ALL_46
    issues: list[str] = []
    sample: list[dict[str, Any]] = []

    missing_required = [c for c in required_id_cols if c not in panel.columns]
    if missing_required:
        issues.append(f"missing required id cols: {missing_required}")
    missing_chars = [c for c in cols if c not in panel.columns]
    if missing_chars:
        issues.append(f"missing characteristic cols: {missing_chars}")

    if "permno" in panel.columns and "date" in panel.columns:
        dup_mask = panel.duplicated(subset=["permno", "date"], keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup:
            issues.append(f"{n_dup:,} duplicate (permno, date) rows")
            sample.extend(
                panel.loc[dup_mask, ["permno", "date"]]
                .head(sample_size)
                .assign(date=lambda d: pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"))
                .to_dict(orient="records")
            )

        not_mono = (
            panel.sort_values(["permno", "date"])
            .groupby("permno")["date"]
            .apply(lambda s: not s.is_monotonic_increasing)
        )
        n_bad_mono = int(not_mono.sum())
        if n_bad_mono:
            issues.append(f"{n_bad_mono:,} permnos with non-monotonic dates")

    for col in cols:
        if col in panel.columns and panel[col].dtype.kind != "f":
            issues.append(f"{col} dtype is {panel[col].dtype}, expected float")
            break

    passed = len(issues) == 0
    n_checked = int(len(panel))
    msg = "schema OK" if passed else "; ".join(issues)
    return AuditResult(
        name="schema",
        passed=passed,
        severity="hard",
        n_checked=n_checked,
        n_violations=len(issues),
        message=msg,
        sample_violations=sample,
    )


def audit_coverage_stability(
    panel: pd.DataFrame,
    *,
    soft_threshold: float = 0.10,
    hard_threshold: float = 0.30,
    known_jumps: tuple[int, ...] = (1986,),
    sample_size: int = 10,
) -> AuditResult:
    """Detect year-over-year firm-count jumps outside known transitions.

    `known_jumps` documents accepted breaks (Doc 3 §"Audit 3.3" — the 1986
    NASDAQ-Compustat integration in particular).

    Returns severity:
    - `"info"` and `passed=True` if max |YoY| <= soft_threshold.
    - `"soft"` and `passed=True` if some months exceed soft but none exceed hard.
    - `"hard"` and `passed=False` if any non-known month exceeds hard_threshold.
    """
    if "permno" not in panel.columns or "date" not in panel.columns:
        return AuditResult(
            name="coverage_stability",
            passed=False,
            severity="hard",
            n_checked=0,
            n_violations=0,
            message="missing permno or date column",
        )

    counts = (
        panel.groupby(panel["date"].dt.to_period("M"))["permno"].nunique().rename("n")
    )
    counts.index = counts.index.to_timestamp("M") + pd.offsets.MonthEnd(0)
    yoy = counts.pct_change(12).rename("yoy")
    df = pd.concat([counts, yoy], axis=1).dropna(subset=["yoy"])

    soft_hits = df[df["yoy"].abs() > soft_threshold]
    hard_hits = df[df["yoy"].abs() > hard_threshold]
    hard_unaccepted = hard_hits[~hard_hits.index.year.isin(known_jumps)]

    n_checked = int(len(df))
    n_soft = int(len(soft_hits))
    n_hard = int(len(hard_unaccepted))
    passed = n_hard == 0
    severity = "hard" if n_hard else ("soft" if n_soft else "info")

    msg_parts = [f"checked {n_checked} months", f"|YoY|>{soft_threshold:.0%}: {n_soft}"]
    if n_hard:
        msg_parts.append(f"unaccepted |YoY|>{hard_threshold:.0%}: {n_hard}")

    # Surface hard violations first when they exist; otherwise show soft hits.
    sample_source = hard_unaccepted if n_hard else soft_hits
    sample = (
        sample_source.head(sample_size)
        .reset_index()
        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m"))
        .to_dict(orient="records")
    )

    return AuditResult(
        name="coverage_stability",
        passed=passed,
        severity=severity,
        n_checked=n_checked,
        n_violations=n_hard,
        message="; ".join(msg_parts),
        sample_violations=sample,
    )


def run_all_audits(
    panel: pd.DataFrame,
    *,
    foundation_dir: Path | None = None,
    tolerance_days: int = 380,
) -> dict[str, AuditResult]:
    """Run all Arm 3 audits and optionally write `arm3_summary.json`.

    Parameters
    ----------
    panel
        Stage 00 final panel.
    foundation_dir
        If provided, write the JSON summary here.
    tolerance_days
        Forwarded to `audit_pit_bounds`.

    Returns
    -------
    dict
        Mapping of audit name -> AuditResult.
    """
    results = {
        "pit_bounds": audit_pit_bounds(panel, tolerance_days=tolerance_days),
        "rank_normalization": audit_rank_normalization(panel),
        "schema": audit_schema(panel),
        "coverage_stability": audit_coverage_stability(panel),
    }
    for r in results.values():
        level = logging.INFO if r.passed else logging.ERROR
        log.log(level, "arm3.%s: %s — %s", r.name, "PASS" if r.passed else "FAIL", r.message)

    if foundation_dir is not None:
        foundation_dir = Path(foundation_dir)
        foundation_dir.mkdir(parents=True, exist_ok=True)
        out_path = foundation_dir / "arm3_summary.json"
        out_path.write_text(
            json.dumps(
                {
                    "all_passed": all(r.passed for r in results.values()),
                    "any_hard_fail": any(
                        (not r.passed) and r.severity == "hard" for r in results.values()
                    ),
                    "audits": {name: r.to_dict() for name, r in results.items()},
                },
                indent=2,
            )
        )
        log.info("Arm 3 summary → %s", out_path)
    return results
