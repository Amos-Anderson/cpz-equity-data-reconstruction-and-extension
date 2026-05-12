"""Stage 00 pipeline orchestrator.

The orchestrator is deliberately plain: it calls the same major steps as the
previous notebook, saving the same intermediate parquet files. Logging is
emitted at INFO so the CLI walks the reader through the flow without needing
to open the notebook.

Acceptance gating
-----------------
After all arms run, `run_stage00` writes a `stage0_acceptance.json` summary
that consolidates the hard-fail status of Arm 2 (FF replication) and Arm 3
(internal audits). Arm 1 is diagnostic only and does not contribute to the
gate (see Doc 3 §"Overview and Acceptance Bar"). The pipeline does not
*raise* on failure --- it logs ERROR-level messages and writes the summary
so the user can inspect every result. Stage 1 should refuse to consume the
panel when `stage0_acceptance.json::stage0_complete == false`.

Flags:

- ``--pull-raw``        : pull raw WRDS files first.
- ``--pull-kf``         : pull the Ken French extended factor parquet (FF5
                          + UMD + RF) and exit. Use this once before the
                          first Arm 2 run.
- ``--no-validation``   : skip the lightweight CPZ breadth/return summary.
- ``--no-audits``       : skip the Arm 3 internal audits.
- ``--no-arm1``         : skip the Arm 1 distributional CPZ comparison.
- ``--no-arm2``         : skip the Arm 2 Fama-French replication.
- ``--no-figures``      : skip plotly+PNG figure generation.
- ``--skip-build``      : reuse on-disk intermediates; only run assembly +
                          validation. Useful for re-running audits without
                          re-doing the expensive SUV / 252-day work.
- ``--verbose``         : DEBUG-level logging.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from data_reconstruction.accounting import build_accounting_file
from data_reconstruction.assemble import assemble_panel
from data_reconstruction.config import Stage00Config
from data_reconstruction.crsp import build_clean_monthly
from data_reconstruction.download import pull_all_raw_data
from data_reconstruction.monthly import build_monthly_file
from data_reconstruction.risk import build_risk_file
from data_reconstruction.validation import (
    annual_breadth_table,
    annual_returns_table,
    char_stats_table,
    compare_to_cpz,
    compare_to_kf,
    load_kf_factors,
    pull_kf_extended_standalone,
    run_all_audits,
    validate_breadth_and_returns,
)
from data_reconstruction.validation import diagnostics as fig_helpers

log = logging.getLogger("data_reconstruction.pipeline")


def _configure_logging(verbose: bool = False) -> None:
    """Idempotent logging configuration for the package.

    Tests and library consumers can call this with verbose=True; if a
    handler is already attached, level is updated in place.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger("data_reconstruction")
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.propagate = False


def run_stage00(
    config: Stage00Config | None = None,
    *,
    pull_raw: bool = False,
    skip_build: bool = False,
    run_validation: bool = True,
    run_audits: bool = True,
    run_arm1: bool = True,
    run_arm2: bool = True,
    write_figures: bool = True,
) -> pd.DataFrame:
    """Run Stage 00 from raw WRDS files to final ranked panel.

    Parameters
    ----------
    config
        Stage 00 paths and parameters.
    pull_raw
        If True, pull raw WRDS files before local construction.
    skip_build
        If True, skip every ``build_*`` step and only run the final
        ``assemble_panel`` + validation. Use this to re-run audits or
        diagnostics against existing intermediates without redoing the
        slow steps. The factor panel is regenerated from the existing
        ``monthly_chars.parquet``, ``risk_chars.parquet``,
        ``accounting_chars.parquet`` on disk.
    run_validation
        If True, write the lightweight CPZ breadth/return summary.
    run_audits
        If True, run the Arm 3 internal audits and write ``arm3_summary.json``.
    run_arm1
        If True, run the Arm 1 distributional CPZ comparison and write the
        four ``arm1_*.parquet`` artifacts plus ``arm1_summary.json``.
    run_arm2
        If True, run the Arm 2 FF5+UMD replication vs Ken French and write
        ``arm2_factor_returns.parquet``, ``arm2_per_factor_summary.parquet``,
        and ``arm2_summary.json``.
    write_figures
        If True, render plotly+PNG figures for coverage, Arm 1 and Arm 2
        outputs to ``config.figures_dir``.
    """
    config = config or Stage00Config()
    config.ensure_dirs()
    log.info(
        "stage00 start: raw_dir=%s foundation_dir=%s sample=%s..%s tolerance=%dd",
        config.raw_dir,
        config.foundation_dir,
        config.sample_start,
        config.sample_end,
        config.pit_tolerance_days,
    )

    if pull_raw:
        pull_all_raw_data(config)

    if skip_build:
        log.info("stage00: --skip-build set; skipping builders, using existing intermediates")
    else:
        build_clean_monthly(config)
        build_accounting_file(config)
        build_monthly_file(config)
        build_risk_file(config)

    panel = assemble_panel(config)

    if run_audits and len(panel) > 0:
        run_all_audits(
            panel,
            foundation_dir=config.foundation_dir,
            tolerance_days=config.pit_tolerance_days,
        )

    if run_validation and config.cpz_reference_path.exists() and len(panel) > 0:
        validate_breadth_and_returns(panel, config)

    if run_arm1 and config.cpz_reference_path.exists() and len(panel) > 0:
        compare_to_cpz(
            panel,
            config.cpz_reference_path,
            config.foundation_dir,
        )

    if run_arm2 and len(panel) > 0:
        try:
            kf = load_kf_factors(config.raw_dir)
            compare_to_kf(panel, kf, config.foundation_dir)
        except FileNotFoundError as exc:
            log.warning("arm2 skipped: %s", exc)

    if write_figures and len(panel) > 0:
        _write_figures(panel, config)

    if len(panel) > 0:
        _write_stage0_acceptance(config)

    log.info(
        "stage00 complete: %d rows, %d firms",
        len(panel),
        int(panel["permno"].nunique()) if len(panel) else 0,
    )
    return panel


def _write_stage0_acceptance(config: Stage00Config) -> dict:
    """Consolidate Arm 2 + Arm 3 status into `stage0_acceptance.json`.

    Two gates are reported:

    - ``all_arms_mechanically_passed`` — strict: every Arm 2 and Arm 3
      criterion meets its threshold with no exceptions. This is the raw
      empirical result.
    - ``stage0_complete`` — the headline: True iff every failure sits
      within the *accepted caveat* lists in `Stage00Config`. The default
      caveats encode the universe-attributable failures documented in
      DECISION_LOG entry 041 (financials excluded; Arm 3 coverage
      stability 1984 step-up; Arm 2 amplitude failures for HML, RMW,
      UMD). Both flags are reported so the mechanical result is never
      hidden.
    """
    foundation_dir = config.foundation_dir
    summary: dict = {
        "stage0_complete": False,
        "all_arms_mechanically_passed": False,
        "arm3_passed": None,
        "arm3_hard_failures": [],
        "arm2_passed": None,
        "arm2_failing_factors": {},
        "arm1_severity": "diagnostic",
        "arm1_breadth_max_abs_pct_diff": None,
        "accepted_caveats": {"arm3": [], "arm2": []},
        "unaccepted_failures": {"arm3": [], "arm2": {}},
    }

    arm3_path = foundation_dir / "arm3_summary.json"
    if arm3_path.exists():
        a3 = json.loads(arm3_path.read_text())
        summary["arm3_passed"] = bool(a3.get("all_passed", False)) and not bool(
            a3.get("any_hard_fail", True)
        )
        summary["arm3_hard_failures"] = [
            name
            for name, r in a3.get("audits", {}).items()
            if (not r.get("passed", False)) and r.get("severity") == "hard"
        ]

    arm2_path = foundation_dir / "arm2_summary.json"
    if arm2_path.exists():
        a2 = json.loads(arm2_path.read_text())
        summary["arm2_passed"] = bool(a2.get("arm2_passed", False))
        summary["arm2_failing_factors"] = {
            name: c.get("failing", [])
            for name, c in a2.get("acceptance", {}).get("criteria", {}).items()
            if c.get("failing")
        }

    arm1_path = foundation_dir / "arm1_summary.json"
    if arm1_path.exists():
        a1 = json.loads(arm1_path.read_text())
        summary["arm1_breadth_max_abs_pct_diff"] = a1.get("breadth", {}).get(
            "max_abs_pct_diff"
        )

    # Mechanical gate: True iff there are zero failures at the strict thresholds.
    summary["all_arms_mechanically_passed"] = bool(
        summary["arm3_passed"] and summary["arm2_passed"]
    )

    # Caveat-aware gate: partition failures into accepted vs unaccepted.
    arm3_caveats = set(config.accepted_arm3_caveats or [])
    arm2_caveats = set(config.accepted_arm2_caveats or [])

    arm3_accepted = [c for c in summary["arm3_hard_failures"] if c in arm3_caveats]
    arm3_unaccepted = [c for c in summary["arm3_hard_failures"] if c not in arm3_caveats]

    arm2_accepted: list[str] = []
    arm2_unaccepted: dict[str, list[str]] = {}
    for criterion, factors in summary["arm2_failing_factors"].items():
        for f in factors:
            key = f"{f}.{criterion}"
            if key in arm2_caveats:
                arm2_accepted.append(key)
            else:
                arm2_unaccepted.setdefault(criterion, []).append(f)

    summary["accepted_caveats"] = {"arm3": arm3_accepted, "arm2": arm2_accepted}
    summary["unaccepted_failures"] = {"arm3": arm3_unaccepted, "arm2": arm2_unaccepted}

    # Stage 0 complete iff no unaccepted failures remain.
    summary["stage0_complete"] = (not arm3_unaccepted) and (not arm2_unaccepted)

    out_path = foundation_dir / "stage0_acceptance.json"
    out_path.write_text(json.dumps(summary, indent=2))

    if summary["stage0_complete"]:
        headline = (
            "PASS (mechanically clean)"
            if summary["all_arms_mechanically_passed"]
            else "PASS (with accepted caveats)"
        )
        level = logging.INFO
    else:
        headline = "FAIL (unaccepted failures present)"
        level = logging.ERROR
    log.log(
        level,
        "stage0 acceptance: %s | mechanical: arm2=%s arm3=%s | accepted_caveats=%d -> %s",
        headline,
        summary["arm2_passed"],
        summary["arm3_passed"],
        len(arm3_accepted) + len(arm2_accepted),
        out_path,
    )
    return summary


def _write_figures(panel: pd.DataFrame, config: Stage00Config) -> None:
    """Render coverage and Arm 1 plotly figures to `config.figures_dir`."""
    foundation_dir = config.foundation_dir
    figures_dir = config.figures_dir

    # Coverage time-series figure (uses our panel)
    fig = fig_helpers.coverage_stability_figure(panel)
    fig_helpers.save_figure(fig, "panel_coverage", figures_dir)

    # Pre-filter characteristic presence
    diag_csv = foundation_dir / "completeness_diag.csv"
    if diag_csv.exists():
        diag = pd.read_csv(diag_csv)
        fig = fig_helpers.char_presence_figure(diag)
        fig_helpers.save_figure(fig, "char_presence", figures_dir)

    # Arm 1 figures (only if Arm 1 artifacts exist)
    arm1_files = {
        "arm1_annual_breadth": foundation_dir / "arm1_annual_breadth.parquet",
        "arm1_annual_returns": foundation_dir / "arm1_annual_returns.parquet",
        "arm1_char_stats": foundation_dir / "arm1_char_stats.parquet",
        "arm1_yearly_coverage": foundation_dir / "arm1_yearly_coverage.parquet",
    }
    if arm1_files["arm1_annual_breadth"].exists():
        fig = fig_helpers.arm1_annual_breadth_figure(
            pd.read_parquet(arm1_files["arm1_annual_breadth"])
        )
        fig_helpers.save_figure(fig, "arm1_annual_breadth", figures_dir)
    if arm1_files["arm1_annual_returns"].exists():
        fig = fig_helpers.arm1_annual_returns_figure(
            pd.read_parquet(arm1_files["arm1_annual_returns"])
        )
        fig_helpers.save_figure(fig, "arm1_annual_returns", figures_dir)
    if arm1_files["arm1_char_stats"].exists():
        fig = fig_helpers.arm1_char_stats_figure(
            pd.read_parquet(arm1_files["arm1_char_stats"])
        )
        fig_helpers.save_figure(fig, "arm1_char_stats", figures_dir)
    if arm1_files["arm1_yearly_coverage"].exists():
        fig = fig_helpers.arm1_yearly_coverage_figure(
            pd.read_parquet(arm1_files["arm1_yearly_coverage"])
        )
        fig_helpers.save_figure(fig, "arm1_yearly_coverage", figures_dir)

    arm2_returns_path = foundation_dir / "arm2_factor_returns.parquet"
    arm2_diag_path = foundation_dir / "arm2_per_factor_summary.parquet"
    if arm2_returns_path.exists():
        fig = fig_helpers.arm2_cumulative_returns_figure(pd.read_parquet(arm2_returns_path))
        fig_helpers.save_figure(fig, "arm2_cumulative_returns", figures_dir)
    if arm2_diag_path.exists():
        fig = fig_helpers.arm2_correlation_summary_figure(pd.read_parquet(arm2_diag_path))
        fig_helpers.save_figure(fig, "arm2_correlation_summary", figures_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 00 data reconstruction.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pull-raw", action="store_true", help="Pull raw WRDS files first.")
    parser.add_argument(
        "--pull-kf",
        action="store_true",
        help="Pull the Ken French extended factor parquet (FF5+UMD+RF) via WRDS and exit.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Reuse on-disk intermediates.")
    parser.add_argument("--no-validation", action="store_true", help="Skip lightweight CPZ summary.")
    parser.add_argument("--no-audits", action="store_true", help="Skip Arm 3 internal audits.")
    parser.add_argument("--no-arm1", action="store_true", help="Skip Arm 1 distributional CPZ comparison.")
    parser.add_argument("--no-arm2", action="store_true", help="Skip Arm 2 FF5+UMD replication.")
    parser.add_argument("--no-figures", action="store_true", help="Skip plotly+PNG figure generation.")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging.")
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    config = Stage00Config.from_yaml(args.config) if args.config else Stage00Config()

    if args.pull_kf:
        pull_kf_extended_standalone(config)
        print(f"Ken French extended factors saved to {config.raw_dir}/ff_factors_monthly_extended.parquet")
        return 0

    panel = run_stage00(
        config,
        pull_raw=args.pull_raw,
        skip_build=args.skip_build,
        run_validation=not args.no_validation,
        run_audits=not args.no_audits,
        run_arm1=not args.no_arm1,
        run_arm2=not args.no_arm2,
        write_figures=not args.no_figures,
    )
    print(f"Stage 00 complete: {len(panel):,} rows, {panel['permno'].nunique():,} firms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
