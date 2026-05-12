"""Command-line entry point for running a pipeline stage.

Usage
-----
    python scripts/run_stage.py --stage 1
    python scripts/run_stage.py --stage 5 --config configs/stage05_risk.yaml
    python scripts/run_stage.py --list

Each stage has a dedicated function that orchestrates the stage's modules.
Stages are implemented incrementally; calling an unimplemented stage raises
NotImplementedError with a pointer to the stage number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Stage registry — maps stage numbers to names and default configs
STAGES: dict[int, dict[str, str]] = {
    1: {"name": "Universe Construction", "config": "configs/stage01_universe.yaml"},
    2: {"name": "Factor Construction", "config": "configs/stage02_factors.yaml"},
    3: {"name": "Factor Validation", "config": "configs/stage03_validation.yaml"},
    4: {"name": "Signal Combination", "config": "configs/stage04_alpha.yaml"},
    5: {"name": "Risk Modeling", "config": "configs/stage05_risk.yaml"},
    6: {"name": "Transaction Cost Model", "config": "configs/stage06_costs.yaml"},
    7: {"name": "Portfolio Construction", "config": "configs/stage07_portfolio.yaml"},
    8: {"name": "Backtest Engine", "config": "configs/stage08_backtest.yaml"},
    9: {"name": "Attribution", "config": "configs/stage09_attribution.yaml"},
    10: {"name": "Alpaca Deployment", "config": "configs/stage10_deployment.yaml"},
    11: {"name": "Monitoring", "config": "configs/stage11_monitoring.yaml"},
}


def run_stage(stage: int, config_path: Path) -> None:
    """Dispatch to the appropriate stage runner.

    Each stage is implemented in its respective notebook / module.
    This CLI is a convenience wrapper for non-interactive runs.
    """
    from factor_pipeline.utils.config import load_yaml_config
    from factor_pipeline.utils.logging import get_logger

    log = get_logger("run_stage")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_yaml_config(config_path)
    log.info("Running Stage %d: %s", stage, STAGES[stage]["name"])
    log.info("Config: %s", config_path)

    if stage == 1:
        _run_stage_1(config)
    elif stage == 2:
        _run_stage_2(config)
    elif stage == 3:
        _run_stage_3(config)
    elif stage == 4:
        _run_stage_4(config)
    elif stage == 5:
        _run_stage_5(config)
    elif stage == 6:
        _run_stage_6(config)
    elif stage == 7:
        _run_stage_7(config)
    elif stage == 8:
        _run_stage_8(config)
    elif stage == 9:
        _run_stage_9(config)
    elif stage == 10:
        _run_stage_10(config)
    elif stage == 11:
        _run_stage_11(config)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def _run_stage_1(config: dict) -> None:
    raise NotImplementedError(
        "Stage 1 entry point — implement after populating factor_pipeline.data"
    )


def _run_stage_2(config: dict) -> None:
    raise NotImplementedError("Stage 2 entry point")


def _run_stage_3(config: dict) -> None:
    raise NotImplementedError("Stage 3 entry point")


def _run_stage_4(config: dict) -> None:
    raise NotImplementedError("Stage 4 entry point")


def _run_stage_5(config: dict) -> None:
    raise NotImplementedError("Stage 5 entry point")


def _run_stage_6(config: dict) -> None:
    raise NotImplementedError("Stage 6 entry point")


def _run_stage_7(config: dict) -> None:
    raise NotImplementedError("Stage 7 entry point")


def _run_stage_8(config: dict) -> None:
    raise NotImplementedError("Stage 8 entry point")


def _run_stage_9(config: dict) -> None:
    raise NotImplementedError("Stage 9 entry point")


def _run_stage_10(config: dict) -> None:
    raise NotImplementedError("Stage 10 entry point")


def _run_stage_11(config: dict) -> None:
    raise NotImplementedError("Stage 11 entry point")


def _list_stages() -> None:
    """Print registered stages to stdout."""
    print("Registered stages:")
    for n, info in sorted(STAGES.items()):
        print(f"  {n:2d}: {info['name']:30s}  config={info['config']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pipeline stage.")
    parser.add_argument("--stage", type=int, help="Stage number (1-11)")
    parser.add_argument("--config", type=Path, help="Override default config path")
    parser.add_argument("--list", action="store_true", help="List registered stages")
    args = parser.parse_args()

    if args.list:
        _list_stages()
        return 0

    if args.stage is None:
        parser.error("--stage is required (or use --list)")

    if args.stage not in STAGES:
        parser.error(f"Unknown stage: {args.stage}. Use --list to see options.")

    config_path = args.config or Path(STAGES[args.stage]["config"])

    try:
        run_stage(args.stage, config_path)
    except NotImplementedError as e:
        print(f"[NotImplementedError] {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())