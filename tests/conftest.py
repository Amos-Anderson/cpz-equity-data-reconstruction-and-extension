"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to project root."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Path to external data directory (gitignored)."""
    env_dir = os.environ.get("ML4T_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / "ml4t_data"


@pytest.fixture(scope="session")
def fixtures_dir(project_root: Path) -> Path:
    """Path to test fixtures (small sample data committed to the repo)."""
    d = project_root / "tests" / "fixtures"
    d.mkdir(parents=True, exist_ok=True)
    return d