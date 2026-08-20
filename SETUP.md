# PropheticIQ — environment setup (Windows / PowerShell)

Run every command from the **project root** (the folder containing `src/`).

---

## 1. Confirm your Python version

```powershell
py -0
```

This lists every Python installed on the machine. You want **3.11 or 3.12**.

Avoid 3.13+ for now: prebuilt wheels for `xgboost`, `shap` and `statsmodels`
lag a few months behind each Python release, and when a wheel is missing pip
falls back to compiling from source — which on Windows fails asking for Visual
Studio Build Tools.

If neither is installed, get 3.12 from python.org and tick
**"Add python.exe to PATH"** during installation.

---

## 2. Create the virtual environment

```powershell
cd C:\path\to\propheticiq
py -3.12 -m venv .venv
```

`py -3.12` selects the interpreter explicitly. Plain `python -m venv` uses
whatever is first on PATH, which on a machine with several Pythons installed
is often not the one you meant.

The folder is named `.venv` because that is the name VS Code detects
automatically, and it is already in `.gitignore`.

> **If your project lives in OneDrive or Dropbox:** move it somewhere local
> (e.g. `C:\dev\propheticiq`). Sync clients continuously scan the thousands of
> small files inside a venv, which makes installs crawl and can corrupt the
> environment mid-write.

---

## 3. Allow PowerShell to run the activation script

This is the step almost everyone hits. By default PowerShell refuses to run
local scripts, so activation fails with:

```
.\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts
is disabled on this system.
```

Fix it once, for your user only:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

`RemoteSigned` allows local scripts while still requiring downloaded scripts
to be signed. `-Scope CurrentUser` means you are not changing machine-wide
security settings and do not need an administrator prompt. Answer `Y` when
asked to confirm.

---

## 4. Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should now be prefixed with `(.venv)`. Verify you are actually
using the venv's interpreter rather than the system one:

```powershell
where.exe python
```

The **first** line must point inside `.venv\Scripts\`. If it does not,
activation did not take effect — do not proceed, or you will install packages
into your system Python.

---

## 5. Upgrade pip, then install

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Upgrade pip first because older versions have a weaker dependency resolver
and occasionally install a mutually incompatible set without complaining.

Use `python -m pip` rather than bare `pip` for the upgrade — on Windows, pip
cannot replace its own running executable, and `python -m pip` avoids the
file-lock error.

Expect 2–5 minutes. This installs everything needed for phases 1 through 14,
including the Streamlit dashboard.

---

## 6. Verify

```powershell
python scripts\check_env.py
```

This checks the Python version, confirms the venv is active, imports every
required package, and counts the CSVs in `data/raw/`. Fix anything it reports
before writing code — "pip said it installed" and "it imports" are different
claims, and the gap between them is where broken wheels hide.

---

## 7. Point VS Code at the environment

1. `Ctrl+Shift+P`
2. Type `Python: Select Interpreter`
3. Choose the one whose path contains `.venv`

Then **close and reopen the integrated terminal** so it picks up the new
interpreter. Without this step VS Code's linter and the terminal can end up on
different Pythons, producing "module not found" errors for packages you
definitely installed.

Recommended `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true,
  "python.analysis.extraPaths": ["."],
  "editor.rulers": [88]
}
```

`python.analysis.extraPaths` set to `.` is what makes `from src.data.load
import ...` resolve in the editor without installing the project as a package.

---

## 8. Run something

```powershell
python scripts\run_eda.py --list
python scripts\run_eda.py A H
```

---

## Optional: transformer NLP (Phase 12 only)

```powershell
pip install -r requirements-nlp.txt
```

**Do not install this yet.** `torch` is roughly 2.5 GB. Build the
TF-IDF + logistic regression sentiment baseline first — it needs only
scikit-learn, which you already have. If the baseline performs acceptably on
53-character Portuguese comments, you may never need this file, and
"I measured, and the simpler model won" is a stronger result than an unused
2.5 GB dependency.

---

## Locking versions for reproducibility

`requirements.txt` uses lower bounds so pip can resolve something that works
on your machine. Once everything runs, capture the exact versions:

```powershell
pip freeze > requirements.lock.txt
```

Commit both. `requirements.txt` states your intent; `requirements.lock.txt`
records what actually worked. Anyone reviewing your portfolio can reproduce
your exact environment from the lock file — a detail that signals real
engineering practice.

---

## Daily workflow

```powershell
cd C:\dev\propheticiq
.\.venv\Scripts\Activate.ps1
# ... work ...
deactivate
```

Steps 1–3 are one-time. Only activation is a daily habit.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Activate.ps1 cannot be loaded` | Execution policy — see step 3 |
| `(.venv)` missing from prompt | Activation failed; check `where.exe python` |
| `ModuleNotFoundError: No module named 'src'` | Not running from the project root. `cd` to the folder containing `src/` |
| `ModuleNotFoundError` for an installed package | VS Code terminal is on a different interpreter. Redo step 7 and reopen the terminal |
| `error: Microsoft Visual C++ 14.0 or greater is required` | pip is compiling from source because no wheel exists for your Python. Use 3.11 or 3.12 |
| Access denied upgrading pip | Use `python -m pip install --upgrade pip`, not bare `pip` |
| Installs extremely slow | Project is inside OneDrive/Dropbox. Move it to a local path |
