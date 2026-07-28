"""
Centralized filesystem paths for the Crypto Alpha Research Laboratory (CARL).

All project modules should import paths from this file rather than
constructing filesystem locations directly.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

CACHE_DIR = DATA_DIR / "cache"

# ---------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TESTS_DIR = PROJECT_ROOT / "tests"

# ---------------------------------------------------------------------
# Ensure required directories exist
# ---------------------------------------------------------------------

_REQUIRED_DIRECTORIES = (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
)

for directory in _REQUIRED_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)