# Deployment

Streamlit Community Cloud, free tier. About 15 minutes.

---

## Why the requirements files are split

Streamlit Cloud clones your repo into a **1GB container** and installs
`requirements.txt`. A single file containing torch would add ~2.5GB and the
build fails or times out.

| File | Contains | Installed on Cloud |
|---|---|---|
| `requirements.txt` | 8 packages the running app imports | ✅ yes |
| `requirements-dev.txt` | xgboost, shap, statsmodels, matplotlib, pytest | ❌ no |
| `requirements-nlp.txt` | torch, transformers | ❌ no |

The deployed churn model is **logistic regression** (scikit-learn), selected on
validation, so xgboost is not needed at runtime. The forecast, recommender,
sentiment and low-review pages read precomputed CSV and parquet output rather
than loading their models.

Verified: all 8 pages run with only `models/churn_model.joblib` present.

---

## Step 1 — check what will be committed

```powershell
git status --short
git check-ignore -v data/raw/olist_orders_dataset.csv
```

The second command must print a line — that confirms the 121MB of raw CSVs are
ignored. If it prints nothing, `.gitignore` is not being applied.

Expect roughly **51MB** committed:

| Kept | Why |
|---|---|
| `data/processed/*` (~32MB) | The app cannot start without it — there is no build step on Cloud |
| `models/churn_model.joblib` (3MB) | Page 4 scores live |
| `data/processed/*_comparison.csv` | Model Performance page reads them |
| all code, notebooks, tests, figures | |

| Excluded | Why |
|---|---|
| `data/raw/*` (121MB) | Redistributable from Kaggle |
| the other three `.joblib` files (59MB) | Training artifacts; the app reads their output |
| `.venv/`, `__pycache__/` | Environment |

---

## Step 2 — initialise and push

```powershell
git init
git add .
git commit -m "PropheticIQ: end-to-end e-commerce decision intelligence platform"
git branch -M main
git remote add origin https://github.com/YOURNAME/PropheticIQ.git
git push -u origin main
```

Create the repo on github.com first — **public**, and do not let GitHub add a
README or .gitignore (you already have both).

If the push is rejected for size, something in `data/raw/` slipped through:

```powershell
git rm -r --cached data/raw
git commit -m "Remove raw data from tracking"
```

---

## Step 3 — deploy

1. Go to **share.streamlit.io**
2. Sign in with GitHub, authorise access
3. **New app** → **Deploy a public app from GitHub**
4. Fill in:
   - Repository: `YOURNAME/PropheticIQ`
   - Branch: `main`
   - **Main file path: `app/Home.py`**
5. **Advanced settings → Python version: 3.12**
6. **Deploy**

First build takes 3–5 minutes. Watch the log panel; if it fails, the traceback
appears there.

Your URL: `https://YOURNAME-propheticiq-app-home-xxxxx.streamlit.app`

Rename it under **Settings → General → App URL** to something like
`propheticiq.streamlit.app` if free.

---

## Step 4 — verify the live app

- [ ] Executive Overview loads with 4 KPIs
- [ ] Sidebar shows all 8 pages
- [ ] Churn page: drag the budget slider — it should respond instantly
      (scores are cached; only the threshold recomputes)
- [ ] Product Recommendations: pick a seed, get 10 results
- [ ] Review Intelligence: theme heatmap renders
- [ ] Model Performance: five expanders open, figures display
- [ ] Churn page footer says the deployed model is **logistic** (if it says
      random_forest, re-run `notebooks/04_churn_model.py` and push)

If a page shows a **missing artifact** table instead of content, the listed
file was not committed. Check `.gitignore` did not exclude it.

---

## Step 5 — put the URL in the README

Line 8 of `README.md`:

```markdown
**Live dashboard:** https://propheticiq.streamlit.app
```

Add a screenshot immediately below it — most people will not click the link,
but everyone sees an image.

```powershell
mkdir docs\images
# save a screenshot as docs/images/dashboard.png, then:
```

```markdown
![PropheticIQ dashboard](docs/images/dashboard.png)
```

```powershell
git add README.md docs/images
git commit -m "Add live URL and screenshot"
git push
```

Streamlit Cloud redeploys automatically on every push.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError` in the build log | Package missing from `requirements.txt`. Add it, push. |
| `FileNotFoundError: dashboard_orders.parquet` | `data/processed/` was gitignored. Check the `!data/processed/` negation lines. |
| `InconsistentVersionWarning` on unpickle | scikit-learn version drift. `requirements.txt` pins `>=1.6,<2.0`; models were trained on 1.8. |
| App boots then dies | 1GB memory limit exceeded. Reduce `n_tail` in `score_churn_cached()`. |
| Build times out | Something heavy is in `requirements.txt` — confirm torch is not there. |
| Pages show "missing artifact" | That file was not committed. Run the command the table names, then push. |

---

## Resume format

```
PropheticIQ — E-Commerce Decision Intelligence Platform
github.com/YOURNAME/PropheticIQ  ·  propheticiq.streamlit.app
```

Lead bullets with findings, not the tech list:

> Built an end-to-end analytics platform on 100k Brazilian marketplace orders:
> customer segmentation (K-Means), churn and low-review prediction (XGBoost +
> SHAP, 3.19x PR-AUC lift), 30-day revenue forecasting (SARIMA, 13% better than
> a seasonal-naive baseline), content-based recommendations (recall@10 0.22,
> 734x random), and Portuguese review sentiment (F1 0.88). Deployed as an
> 8-page Streamlit dashboard with cached inference and a live campaign-ROI tool.

> Quantified where models do not work rather than overselling: proved
> collaborative filtering infeasible at 99.996% matrix sparsity, and reported
> the churn model's 1.32x lift honestly rather than hiding it behind the 98%
> accuracy a do-nothing baseline achieves. Isolated the cause with a controlled
> comparison -- the same pipeline reaches 3.19x lift on a second target, so the
> limit is the data, not the method.

The second bullet is the differentiator. Interviewers see many Olist projects
claiming great results; almost none report what failed and why.