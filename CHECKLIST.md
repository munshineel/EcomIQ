# PropheticIQ — setup checklist

Tick each box in order. Do not skip ahead: every step depends on the one
before it, and the failure modes are much harder to diagnose out of sequence.

---

## Phase 0 — Prerequisites

- [ ] **Python 3.11 or 3.12 installed.** Check with `py -0`.
      Avoid 3.13+: wheels for `xgboost`, `shap` and `statsmodels` lag each new
      Python release, and a missing wheel means pip compiles from source, which
      on Windows fails asking for Visual Studio Build Tools.
- [ ] **Project folder is on a local drive**, not inside OneDrive or Dropbox.
      Sync clients continuously scan the thousands of small files in a venv,
      which makes installs crawl and can corrupt the environment mid-write.
      Recommended: `C:\dev\propheticiq`.
- [ ] **Git installed** (`git --version`) if you plan to publish this.

---

## Phase 1 — Virtual environment

```powershell
cd C:\dev\propheticiq
py -3.12 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

- [ ] `.venv\` folder exists in the project root
- [ ] Prompt shows the `(.venv)` prefix
- [ ] `where.exe python` returns a path inside `.venv\Scripts\` on its
      **first** line

> If activation is blocked with *"running scripts is disabled on this
> system"*, the `Set-ExecutionPolicy` line above is the fix. `RemoteSigned`
> still requires downloaded scripts to be signed, and `-Scope CurrentUser`
> means no admin prompt and no machine-wide change.

If the third box fails, stop. Installing now would put packages into your
system Python, and you would spend an hour on phantom import errors.

---

## Phase 2 — Install

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- [ ] pip upgraded without an access-denied error
      (use `python -m pip`, never bare `pip`, for this one command — pip
      cannot replace its own running executable on Windows)
- [ ] Install completed with no red `ERROR` lines
- [ ] `pip list` shows pandas, scikit-learn, xgboost, shap, statsmodels,
      streamlit

**Do NOT install `requirements-nlp.txt` yet.**

---

## Phase 3 — Project structure

- [ ] Folder tree matches:

```
propheticiq/
├── config/config.yaml
├── data/{raw,interim,processed}/
├── models/
├── notebooks/
├── reports/figures/
├── scripts/{run_eda.py, check_env.py}
├── src/
│   ├── config.py
│   ├── data/{schema.py, load.py, prepare.py}
│   ├── eda/{customer_behaviour, sales_revenue, products, categories,
│   │        time_trends, pricing_payments, reviews, retention}.py
│   ├── features/  models/  evaluation/  viz/
├── tests/
├── .gitignore
├── requirements.txt
├── requirements-nlp.txt
└── SETUP.md
```

- [ ] Every package folder under `src/` contains an `__init__.py`
      (including `src/` itself) — without these, `python -m src.eda.reviews`
      fails with `No module named src`
- [ ] `.gitignore` excludes `.venv/`, `data/raw/*`, `models/*`

---

## Phase 4 — Data

- [ ] All 9 Olist CSVs are in `data/raw/`:

```
olist_customers_dataset.csv          olist_orders_dataset.csv
olist_geolocation_dataset.csv        olist_products_dataset.csv
olist_order_items_dataset.csv        olist_sellers_dataset.csv
olist_order_payments_dataset.csv     product_category_name_translation.csv
olist_order_reviews_dataset.csv
```

- [ ] Filenames are unchanged from the download — `src/data/schema.py` matches
      on exact filenames and will fail loudly rather than guess
- [ ] Data is **not** committed to git

---

## Phase 5 — Verify

```powershell
python scripts\check_env.py
```

- [ ] Python version: OK
- [ ] Virtual environment active: OK
- [ ] All required packages import (not merely "pip said it installed" —
      importing catches broken or wrong-architecture wheels)
- [ ] Raw data: 9/9 CSVs found
- [ ] Exit code 0

---

## Phase 6 — VS Code

1. `Ctrl+Shift+P` → `Python: Select Interpreter` → the one containing `.venv`
2. **Close and reopen the integrated terminal**

- [ ] Interpreter shown in the status bar points at `.venv`
- [ ] A fresh terminal shows `(.venv)` automatically

`.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true,
  "python.analysis.extraPaths": ["."],
  "editor.rulers": [88]
}
```

`python.analysis.extraPaths: ["."]` is what makes `from src.data.load import
...` resolve in the editor without installing the project as a package.

> Skipping the terminal reopen is the most common cause of
> `ModuleNotFoundError` for a package you can see in `pip list`: the linter
> and the terminal end up on different interpreters.

---

## Phase 7 — Smoke test

```powershell
python scripts\run_eda.py --list
python scripts\run_eda.py A H
python scripts\run_eda.py
```

- [ ] `--list` prints 8 sections
- [ ] `A H` completes and writes 5 PNGs to `reports/figures/eda/`
- [ ] Full run completes and `reports/figures/eda/` contains **20** PNGs
- [ ] Printed numbers match: 99,441 orders / 94,986 customers /
      1.92% corrected repeat rate

If the repeat rate is 3.04% rather than 1.92%, the basket-split correction is
not being applied — check `repeat_gap_days` in `build_customer_frame`.

---

## Phase 8 — Lock the environment

```powershell
pip freeze > requirements.lock.txt
git add requirements.txt requirements.lock.txt
```

- [ ] `requirements.lock.txt` exists and is committed

Commit both files. `requirements.txt` declares intent with lower bounds;
`requirements.lock.txt` records the exact versions that worked. A reviewer can
reproduce your environment precisely from the lock file — a small detail that
signals real engineering practice.

---

## Daily workflow

```powershell
cd C:\dev\propheticiq
.\.venv\Scripts\Activate.ps1
# ... work ...
deactivate
```

Phases 0-3 and 6 are one-time. Only activation is a daily habit.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Activate.ps1 cannot be loaded` | Execution policy — see Phase 1 |
| No `(.venv)` in prompt | Activation failed; check `where.exe python` |
| `No module named 'src'` | Not in the project root, or a missing `__init__.py` |
| `ModuleNotFoundError` for an installed package | VS Code terminal on a different interpreter — redo Phase 6 |
| `Microsoft Visual C++ 14.0 or greater is required` | No wheel for your Python; use 3.11 or 3.12 |
| Access denied upgrading pip | Use `python -m pip install --upgrade pip` |
| `FileNotFoundError: Expected olist_*.csv` | Filename changed, or files are in a subfolder of `data/raw/` |
| Installs extremely slow | Project is inside OneDrive/Dropbox — move it |
