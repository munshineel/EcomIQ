"""Verify the environment is correctly set up.

Run this immediately after installing requirements, before writing any code:

    python scripts/check_env.py

It checks three things that commonly go wrong:
  1. Python version is one the ML wheels actually support.
  2. Every required package imports (not just "pip said it installed").
  3. You are running inside the venv, not the system Python.

Exit code is 0 if everything passes, 1 otherwise, so it also works in CI.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# (import name, pip name) — these differ often enough to be worth listing.
REQUIRED = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("yaml", "pyyaml"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("shap", "shap"),
    ("statsmodels", "statsmodels"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("streamlit", "streamlit"),
    ("pytest", "pytest"),
]

OPTIONAL = [("torch", "torch"), ("transformers", "transformers")]

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 13)  # exclusive upper bound


def check_python() -> bool:
    v = sys.version_info
    ok = MIN_PYTHON <= (v.major, v.minor) < MAX_PYTHON
    flag = "OK  " if ok else "WARN"
    print(f"[{flag}] Python {v.major}.{v.minor}.{v.micro}")
    if not ok:
        print(
            f"       Recommended: 3.11 or 3.12. Versions >= "
            f"{MAX_PYTHON[0]}.{MAX_PYTHON[1]} often lack prebuilt ML wheels."
        )
    return ok


def check_venv() -> bool:
    # sys.prefix differs from sys.base_prefix only inside a virtual env.
    in_venv = sys.prefix != sys.base_prefix
    print(f"[{'OK  ' if in_venv else 'FAIL'}] Virtual environment active")
    if in_venv:
        print(f"       {sys.prefix}")
    else:
        print("       You are on the system Python. Activate .venv first.")
    return in_venv


def check_packages(packages: list[tuple[str, str]], label: str) -> list[str]:
    print(f"\n{label}")
    missing: list[str] = []
    for import_name, pip_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK  ] {pip_name:<20} {version}")
        except ImportError:
            print(f"  [FAIL] {pip_name:<20} not importable")
            missing.append(pip_name)
    return missing


def check_data() -> bool:
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    csvs = sorted(p.name for p in raw.glob("*.csv")) if raw.exists() else []
    ok = len(csvs) == 9
    print(f"\n[{'OK  ' if ok else 'WARN'}] Raw data: {len(csvs)}/9 CSV files in data/raw/")
    if not ok:
        print("       Download the Olist dataset and unzip it into data/raw/")
    return ok


def main() -> int:
    print("=" * 62)
    print("  PropheticIQ environment check")
    print("=" * 62)

    py_ok = check_python()
    venv_ok = check_venv()
    missing = check_packages(REQUIRED, "Required packages:")
    check_packages(OPTIONAL, "Optional packages (Phase 12 transformer NLP):")
    data_ok = check_data()

    print("\n" + "=" * 62)
    if missing:
        print("  FAILED — install the missing packages:")
        print(f"    pip install {' '.join(missing)}")
        return 1
    if not venv_ok:
        print("  FAILED — activate the virtual environment and re-run.")
        return 1
    if not py_ok or not data_ok:
        print("  PASSED with warnings (see above).")
        return 0
    print("  All checks passed. Ready to run: python scripts/run_eda.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())