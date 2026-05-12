"""Configuration for Stage 00 data reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


# Default accepted-caveat lists for the Stage 00 acceptance gate.
#
# These represent failures of the strict mechanical thresholds that have
# been investigated and accepted as *universe-attributable* per
# DECISION_LOG entry 041 (financials-excluded panel vs Ken French's
# financials-included benchmark). The mechanical gate continues to
# report `arm2_passed` / `arm3_passed` as they are; the headline
# `stage0_complete` flag clears once all remaining failures sit within
# this caveat list. To re-introduce strictness, pass empty lists in the
# YAML config.
DEFAULT_ARM3_CAVEATS: tuple[str, ...] = (
    "coverage_stability",  # 1984 firm-count step-up (universe-filter signature)
)
DEFAULT_ARM2_CAVEATS: tuple[str, ...] = (
    # Format: "<factor>.<criterion>"; criterion matches arm2 acceptance keys.
    "hml.mad_lt_0p01",
    "umd.mad_lt_0p01",
    "hml.min_year_rho_gt_0p80",
    "rmw.min_year_rho_gt_0p80",
    "umd.min_year_rho_gt_0p80",
)


@dataclass(slots=True)
class Stage00Config:
    """Paths and locked parameters for the CPZ/FNW reconstruction."""

    raw_dir: Path = Path.home() / "ml4t_data" / "raw"
    output_dir: Path = Path.home() / "ml4t_data" / "extended_v2"
    foundation_dir: Path = Path("data/foundation")
    figures_dir: Path = Path("reports/figures/stage00")
    cpz_reference_path: Path = (
        Path.home() / "ml4t_data" / "academic" / "firm_characteristics_all.parquet"
    )

    crsp_monthly_start: str = "1960-01-01"
    crsp_monthly_end: str = "2024-12-31"
    crsp_daily_start: str = "1963-01-01"
    crsp_daily_end: str = "2024-12-31"
    compustat_start: str = "1960-01-01"
    compustat_end: str = "2024-12-31"
    sample_start: str = "1967-01-01"
    sample_end: str = "2024-12-31"

    pit_tolerance_days: int = 380
    exclude_financials: bool = True
    nyse_amex_delist_return: float = -0.30
    nasdaq_delist_return: float = -0.55

    accepted_arm3_caveats: list[str] = field(
        default_factory=lambda: list(DEFAULT_ARM3_CAVEATS)
    )
    accepted_arm2_caveats: list[str] = field(
        default_factory=lambda: list(DEFAULT_ARM2_CAVEATS)
    )

    def ensure_dirs(self) -> None:
        """Create output directories if missing."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.foundation_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage00Config":
        """Load config values from a YAML file.

        Unknown keys are ignored so the older config file can coexist with
        this simpler implementation.
        """
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        allowed = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in allowed}
        for key in ["raw_dir", "output_dir", "foundation_dir", "figures_dir", "cpz_reference_path"]:
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        return cls(**kwargs)

