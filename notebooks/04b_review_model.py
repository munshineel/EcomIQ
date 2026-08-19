# %% [markdown]
# # 04b - Low review score model
#
# Second target, same machinery: predict a **1-2 star review** at order time.
#
# | | Churn | Low review |
# |---|---|---|
# | Grain | customer | order |
# | Rows | 57,049 | 95,568 |
# | Base rate | 1.8% | 12.8% |
# | `month_index` | excluded (test months unseen) | **included** (same month range) |
# | Review-derived features | allowed | **excluded** (circular) |
#
# The comparison is the point: identical pipeline, one weak result and one
# strong one, explained by properties of the data rather than of the modelling.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "src").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("project root :", ROOT)

# %%
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

import src.models.review as R
from src.config import RANDOM_SEED
from src.models.churn import (
    evaluate, explain, make_models, time_split_three, tune_threshold,
)
from src.viz.style import PALETTE, apply_style

apply_style()
pd.set_option("display.width", 200)

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIG = ROOT / "reports" / "figures" / "review_model"
MODELS.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

def show(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig

# %% [markdown]
# ## Load and check for leakage
#
# `review_score`, `has_comment` and `comment_length` are all properties of the
# review itself. Using them to predict the review's score would be circular.

# %%
path = PROCESSED / "review_features.parquet"
reviews = (pd.read_parquet(path) if path.exists()
           else pd.read_csv(PROCESSED / "review_features.csv", parse_dates=[R.TS_COL]))

FEATURES = R.NUMERIC_FEATURES + R.CATEGORICAL_FEATURES
R.assert_no_leakage(FEATURES)
print(f"{len(reviews):,} orders | {len(FEATURES)} features | "
      f"positives {reviews[R.TARGET].sum():,} ({reviews[R.TARGET].mean():.2%})")

# %% [markdown]
# ## Temporal split

# %%
train, val, test = time_split_three(reviews, val_size=0.2, test_size=0.2,
                                    target=R.TARGET)
print(pd.DataFrame([
    {"split": n, "rows": len(d), "start": d[R.TS_COL].min().date(),
     "end": d[R.TS_COL].max().date(), "positives": int(d[R.TARGET].sum()),
     "rate": round(d[R.TARGET].mean(), 4)}
    for n, d in [("train", train), ("val", val), ("test", test)]
]).set_index("split").to_string())

# %% [markdown]
# Note the test rate is LOWER than train (~9.7% vs ~13.6%). That is the E14
# delivery drift: performance genuinely improved through 2018, so the test
# period has fewer bad reviews. `month_index` is in the feature set to absorb
# it. Test scores will look worse partly for a legitimate reason.

# %% [markdown]
# ## Train

# %%
scale_pos_weight = (train[R.TARGET] == 0).sum() / (train[R.TARGET] == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.2f}  (vs ~50 for churn)")

models = make_models(scale_pos_weight, R.NUMERIC_FEATURES, R.CATEGORICAL_FEATURES)
proba_val, proba_test, thresholds = {}, {}, {}

for name, model in models.items():
    model.fit(train[FEATURES], train[R.TARGET])
    proba_val[name] = model.predict_proba(val[FEATURES])[:, 1]
    proba_test[name] = model.predict_proba(test[FEATURES])[:, 1]
    thresholds[name] = tune_threshold(val[R.TARGET].to_numpy(), proba_val[name])
    print(f"  {name:14s} threshold from validation = {thresholds[name]:.4f}")

# %% [markdown]
# ## Select on validation, report on test

# %%
validation = pd.DataFrame([
    {**evaluate(val[R.TARGET].to_numpy(), proba_val[n], thresholds[n]), "model": n}
    for n in models
]).set_index("model")[["pr_auc", "pr_auc_lift", "roc_auc", "precision", "recall", "f1"]]
print("VALIDATION (selection)")
print(validation.round(4).to_string())

best_name = validation["pr_auc"].idxmax()
best_model = models[best_name]
print(f"\nselected: {best_name}")

# %%
comparison = pd.DataFrame([
    {**evaluate(test[R.TARGET].to_numpy(), proba_test[n], thresholds[n]), "model": n}
    for n in models
]).set_index("model")[["pr_auc", "pr_auc_lift", "roc_auc", "precision", "recall",
                       "f1", "accuracy", "tp", "fp", "fn", "flagged", "threshold"]]
print("TEST (reported)")
print(comparison.round(4).to_string())
print(f"\nbase rate: {test[R.TARGET].mean():.4f}")

# %% [markdown]
# ## Churn vs review: the same pipeline on two targets

# %%
churn_comp = pd.read_csv(PROCESSED / "churn_model_comparison.csv", index_col=0)
side_by_side = pd.DataFrame({
    "churn (best)": churn_comp.loc[churn_comp["pr_auc"].idxmax(),
                                   ["pr_auc", "pr_auc_lift", "roc_auc", "f1"]],
    "low review (best)": comparison.loc[best_name,
                                        ["pr_auc", "pr_auc_lift", "roc_auc", "f1"]],
})
print(side_by_side.round(4).to_string())

# %% [markdown]
# **Roughly 2.4x the lift on the same features.** Two reasons, both about data
# rather than modelling:
#
# 1. **Base rate.** 12.8% versus 1.8% means ~12,000 positive examples instead of
#    ~1,000. Rare-event problems are harder at every step.
# 2. **Causal proximity.** Delivery lateness causes a bad review within days.
#    Whether someone returns six months later depends on price comparison,
#    competitor offers and need recurrence -- none of which exist in this data.
#
# This is the honest answer to "why is your churn model weak?": it is not the
# algorithm, it is that Olist records the outcome but not the causes.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, name in enumerate(models):
    p, r, _ = precision_recall_curve(test[R.TARGET], proba_test[name])
    axes[0].plot(r, p, color=PALETTE[i], lw=1.8, label=name)
axes[0].axhline(test[R.TARGET].mean(), color=PALETTE[5], ls="--", label="base rate")
axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
axes[0].set_title("Precision-Recall, low review score"); axes[0].legend()

pred = (proba_test[best_name] >= thresholds[best_name]).astype(int)
ConfusionMatrixDisplay.from_predictions(
    test[R.TARGET], pred, display_labels=["4-5 star", "1-2 star"],
    cmap="Blues", colorbar=False, ax=axes[1])
axes[1].set_title(f"{best_name} @ {thresholds[best_name]:.3f}"); axes[1].grid(False)
fig.tight_layout()
show(fig, "curves_and_matrix")

# %% [markdown]
# ## Operational use: flag at-risk orders before the review arrives
#
# The business value is a pre-emptive service contact, so the useful question is
# "of the orders we flag, how many really do get a bad review?"

# %%
rows = []
for pct in [0.01, 0.02, 0.05, 0.10, 0.20]:
    t = np.quantile(proba_test[best_name], 1 - pct)
    m = evaluate(test[R.TARGET].to_numpy(), proba_test[best_name], t)
    rows.append({"flag_top_pct": f"{pct:.0%}", "orders_flagged": m["flagged"],
                 "bad_reviews_caught": m["tp"], "precision": round(m["precision"], 3),
                 "recall": round(m["recall"], 3),
                 "lift": round(m["precision"] / m["base_rate"], 2)})
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## Explainability
#
# `explain()` dispatches on estimator type -- TreeExplainer for ensembles,
# LinearExplainer for logistic regression -- so this works whichever model
# validation selected.

# %%
prep = best_model.named_steps["prep"]
clf = best_model.named_steps["clf"]
feat_names = list(prep.get_feature_names_out())

sample = test.sample(min(2000, len(test)), random_state=RANDOM_SEED)
X_trans = prep.transform(sample[FEATURES])
background = prep.transform(train[FEATURES].sample(500, random_state=RANDOM_SEED))

import shap
shap_values = explain(clf, background, X_trans)
print(f"SHAP: {shap_values.shape[0]:,} x {shap_values.shape[1]}")

fig = plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_trans, feature_names=feat_names,
                  max_display=15, show=False)
plt.title(f"SHAP - {best_name} (deployed), low review score")
plt.tight_layout()
show(plt.gcf(), "shap_summary")

# %%
mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_names)
print(mean_abs.sort_values(ascending=False).head(15).round(4).to_string())

# %% [markdown]
# ## Save

# %%
joblib.dump({"model": best_model, "model_name": best_name,
             "selected_on": "validation",
             "threshold": thresholds[best_name], "features": FEATURES,
             "target": R.TARGET,
             "test_metrics": comparison.loc[best_name].to_dict()},
            MODELS / "review_model.joblib")
comparison.to_csv(PROCESSED / "review_model_comparison.csv")
validation.to_csv(PROCESSED / "review_model_validation.csv")
print(f"saved models/review_model.joblib ({best_name}, "
      f"threshold {thresholds[best_name]:.4f})")