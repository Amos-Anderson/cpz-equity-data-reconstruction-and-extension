"""Stage 00 validation subpackage.

Three validation arms per `00_data_foundation_doc3_validation.md`:

- Arm 1: CPZ overlap comparison (1967-2016) — `cpz_comparison.py` (Phase 2).
- Arm 2: Fama-French + UMD replication (2017-2024) — `fama_french.py` (Phase 3).
- Arm 3: internal consistency audits — `audits.py` (this phase).

`lightweight.py` retains the breadth/return summary used as a quick sanity
check; it is not part of the formal acceptance bar.

Plotting helpers live in `diagnostics.py`.
"""

from data_reconstruction.validation.arm2 import (
    compare_to_kf,
    evaluate_acceptance,
    per_factor_diagnostics,
)
from data_reconstruction.validation.audits import (
    AuditResult,
    audit_coverage_stability,
    audit_pit_bounds,
    audit_rank_normalization,
    audit_schema,
    run_all_audits,
)
from data_reconstruction.validation.cpz_comparison import (
    annual_breadth_table,
    annual_returns_table,
    char_stats_table,
    compare_to_cpz,
    load_cpz_panel,
)
from data_reconstruction.validation.fama_french import build_ff_factors
from data_reconstruction.validation.ken_french_loader import (
    load_kf_factors,
    pull_kf_extended,
    pull_kf_extended_standalone,
)
from data_reconstruction.validation.lightweight import (
    cpz_target_summary,
    validate_breadth_and_returns,
)

__all__ = [
    "AuditResult",
    "annual_breadth_table",
    "annual_returns_table",
    "audit_coverage_stability",
    "audit_pit_bounds",
    "audit_rank_normalization",
    "audit_schema",
    "build_ff_factors",
    "char_stats_table",
    "compare_to_cpz",
    "compare_to_kf",
    "cpz_target_summary",
    "evaluate_acceptance",
    "load_cpz_panel",
    "load_kf_factors",
    "per_factor_diagnostics",
    "pull_kf_extended",
    "pull_kf_extended_standalone",
    "run_all_audits",
    "validate_breadth_and_returns",
]
