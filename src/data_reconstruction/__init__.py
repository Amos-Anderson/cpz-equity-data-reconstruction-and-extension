"""Stage 00 CPZ/FNW data reconstruction package.

This package is a modular translation of the previous working Stage 00
notebooks.  Each module corresponds to one notebook step: raw WRDS pulls,
clean CRSP returns, accounting characteristics, monthly characteristics,
risk/liquidity characteristics, final assembly, and validation helpers.
"""

from data_reconstruction.config import Stage00Config


def run_stage00(*args, **kwargs):
    """Run the Stage 00 pipeline without importing the orchestrator eagerly."""
    from data_reconstruction.pipeline import run_stage00 as _run_stage00

    return _run_stage00(*args, **kwargs)

__all__ = ["Stage00Config", "run_stage00"]
