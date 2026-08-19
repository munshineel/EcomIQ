"""Project paths and shared constants.

One place that knows where things live. Every other module imports PATHS from
here rather than building its own relative paths, so moving a folder means
editing one file.
"""

from __future__ import annotations

from pathlib import Path

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATHS: dict[str, Path] = {
    "raw": PROJECT_ROOT / "data" / "raw",
    "interim": PROJECT_ROOT / "data" / "interim",
    "processed": PROJECT_ROOT / "data" / "processed",
    "models": PROJECT_ROOT / "models",
    "reports": PROJECT_ROOT / "reports",
    "figures": PROJECT_ROOT / "reports" / "figures",
}

RANDOM_SEED = 42

# Usable analysis window. Outside this range Olist is too sparse to model:
# November 2016 has zero orders, and September-October 2018 has 20 combined.
ANALYSIS_START = "2017-01-01"
ANALYSIS_END = "2018-08-31"

# A second order sooner than this after the first is a basket split (the
# marketplace assigning separate order_ids to one shopping session), not a
# returning customer.
REPEAT_GAP_DAYS = 7


def ensure_dirs() -> None:
    """Create every configured directory if it does not already exist."""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT", "PATHS", "RANDOM_SEED",
    "ANALYSIS_START", "ANALYSIS_END", "REPEAT_GAP_DAYS", "ensure_dirs",
]