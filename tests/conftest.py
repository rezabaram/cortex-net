"""Test configuration."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clean_state_dir():
    """Remove any state directory created during tests."""
    state_dir = Path("./state")
    yield
    if state_dir.exists():
        shutil.rmtree(state_dir)
