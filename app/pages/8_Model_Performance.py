"""Page 8 - Model Performance. Should we trust any of this?"""

import pandas as pd
import streamlit as st

from _shared import (
    FIGURES, action_box, caveat, kpi_row, load_comparison, page_setup,
)

page_setup("Model Performance",
           "Every model, its headline metric, and its honest limitations")

kpi_row([
    ("Churn (logistic)", "1.32× PR-AUC lift", "weak"),
    ("Forecast (SARIMA)", "R$ 4,452 MAE", "13% better than naive"),
    ("Recommender (content)", "0.222 recall@10", "734× random"),
    ("Sentiment (LinearSVC)", "0.876 F1", "strong"),
])
kpi_row([("Low review (XGBoost)", "3.19× PR-AUC lift", "strong")])

st.markdown("---")
st.info("**Read this page before quoting any number above.** One of these five "
        "models is weak, and the reason is a property of the data rather than "
        "of the modelling — the low-review model below uses the same pipeline "
        "and reaches 2.4x the lift.")

MODELS = {
    "Churn / repeat purchase": {
        "file": "churn_model_comparison.csv",
        "metric": "PR-AUC (base rate 0.0162)",
        "verdict": "⚠️ Weak — usable only for narrow targeting",
        "figures": ["churn/curves.png", "churn/error_analysis.png",
                    "churn/shap_summary.png"],
        "notes": [
            "Best model reaches PR-AUC 0.0213 against a 0.0162 base rate — a lift "
            "of just 1.32×, with ROC-AUC 0.55.",
            "The logistic baseline WON model selection on validation, over "
            "Random Forest and XGBoost. When a linear model beats boosted trees, "
            "there is no exploitable non-linear structure to find.",
            "Selection happens on VALIDATION and reporting on test. The two "
            "disagreed — validation chose logistic, test would have chosen "
            "Random Forest — which is precisely the selection bias a held-out "
            "set exists to prevent.",
            "SHAP explains the DEPLOYED model. An earlier version explained "
            "XGBoost while shipping a different model; the explainer now "
            "dispatches on estimator type, since the winner is not always a tree.",
            "Only 1,024 positive examples exist in 57,049 rows.",
            "Accuracy is meaningless here: predicting 'everyone churns' scores "
            "98.4% accuracy with 0% recall.",
            "Missed returners had 14.5 median delivery days versus 9.8 for those "
            "caught — the signal that would explain them is not in this dataset "
            "(no browsing, wishlist or price-comparison data).",
            "Dropping month_index raised Random Forest lift from 1.19× to 1.32×: "
            "test months lie outside the training range and trees cannot "
            "extrapolate a linear trend.",
        ],
    },
    "Low review score": {
        "file": "review_model_comparison.csv",
        "metric": "PR-AUC (base rate 0.097 on the test period)",
        "verdict": "✅ Strong — operationally useful",
        "figures": ["review_model/curves_and_matrix.png",
                    "review_model/shap_summary.png"],
        "notes": [
            "PR-AUC 0.309 against a 0.097 base rate — a 3.19x lift, versus "
            "1.32x for churn on the SAME pipeline and largely the same features.",
            "Flagging the top 1% of orders catches bad reviews with 70.8% "
            "precision (7.33x lift), which supports a pre-emptive service "
            "contact before the review is even written.",
            "Two reasons it beats the churn model, both about data rather than "
            "modelling: a 12.8% base rate gives ~12,000 positives instead of "
            "~1,000, and delivery lateness causes a bad review within days "
            "whereas a repurchase six months out depends on signals Olist does "
            "not record.",
            "SHAP: delivery_days, days_late and n_items dominate — the delivery "
            "cliff from EDA G17, recovered independently by the model.",
            "month_index is KEPT here, unlike the churn model: train and test "
            "share a month range, so the trend is interpolated rather than "
            "extrapolated.",
            "The test positive rate (9.7%) is lower than train (13.6%) because "
            "delivery performance genuinely improved through 2018.",
        ],
    },
    "Sales forecast": {
        "file": "forecast_backtest.csv",
        "metric": "MAE over 5-fold rolling-origin backtest",
        "verdict": "✅ Solid at 30 days, directional at 90",
        "figures": ["forecast/model_comparison.png", "forecast/final_forecast.png",
                    "forecast/residuals.png"],
        "notes": [
            "SARIMA MAE 4,452 versus seasonal-naive 5,127 — 13% better, and the "
            "only model to beat the benchmark.",
            "Both ML lag models LOSE to seasonal naive. Recursive multi-step "
            "forecasting compounds its own error across 30 steps, and Ridge "
            "extrapolates its linear trend feature off the end of the data.",
            "No LSTM: 606 observations against thousands of parameters. The "
            "decision not to fit one is the defensible engineering call.",
            "The 90-day interval crosses zero — honest, since uncertainty "
            "compounds with horizon and 20 months cannot pin down a quarter.",
            "The series is trimmed 6 days: Olist's collection stopped mid-flow, "
            "and the right-censored tail made SARIMA forecast negative revenue.",
        ],
    },
    "Recommender": {
        "file": "recommender_comparison.csv",
        "metric": "recall@10 on 824 held-out co-purchase pairs",
        "verdict": "✅ Strong relative to the data available",
        "figures": ["recommender/model_comparison.png", "recommender/recall_at_k.png"],
        "notes": [
            "Content-based recall@10 = 0.222 against a random baseline of 0.0003 — "
            "734× better, and 3.6× better than collaborative filtering.",
            "Collaborative filtering is infeasible, measurably: 55% of products "
            "sold exactly once, 3.28% of orders hold two distinct products, and "
            "the user-item matrix is 99.9964% empty.",
            "The hybrid weight sweep was monotonic — every reduction in CF weight "
            "improved results, optimum at zero. The best hybrid IS the content "
            "model.",
            "Coverage matters as much as hit rate: pure popularity touches 0.03% "
            "of the catalogue (11 products) versus 17.5% for content-based.",
            "Precision@10 of 0.036 looks low but the ceiling is 0.10 — each query "
            "has exactly one relevant item, so this is 36% of maximum.",
        ],
    },
    "Review sentiment": {
        "file": "sentiment_model_comparison.csv",
        "metric": "F1 on the negative class, chronological hold-out",
        "verdict": "✅ Strong — and the simple baseline won",
        "figures": ["nlp/confusion_matrix.png", "nlp/top_terms.png",
                    "nlp/aspect_sentiment.png"],
        "notes": [
            "TF-IDF + LinearSVC reaches F1 0.876 and ROC-AUC 0.976. Every TF-IDF "
            "variant lands within 1.2 F1 points of the others.",
            "The embedding approach (SVD to 300 dims) is the WORST non-trivial "
            "model at F1 0.845 — on 9-word documents, compressing to latent "
            "topics discards the lexical detail a linear model uses well.",
            "No transformer deployed. ROC-AUC is already 0.976, leaving ~2 points "
            "of headroom against 2.5GB of dependencies and ~1000× slower CPU "
            "inference.",
            "'não' is deliberately kept as a non-stopword: dropping it turns "
            "'não recomendo' into 'recomendo'.",
            "Game of Thrones house names are removed as stopwords — Olist "
            "anonymised partner names that way, and 'lannister' appears 1,208 "
            "times. Without that the model learns an anonymisation artefact.",
            "The corpus over-represents complaints by ~2×, so this measures "
            "complaint themes rather than satisfaction.",
        ],
    },
}

for name, spec in MODELS.items():
    with st.expander(f"**{name}** — {spec['verdict']}", expanded=False):
        st.caption(f"Headline metric: {spec['metric']}")
        try:
            st.dataframe(load_comparison(spec["file"]).round(4), width="stretch")
        except FileNotFoundError:
            st.warning(f"{spec['file']} not found — re-run the relevant notebook.")

        st.markdown("**Limitations and findings**")
        st.markdown("\n".join(f"- {n}" for n in spec["notes"]))

        cols = st.columns(min(3, len(spec["figures"])))
        for col, rel in zip(cols, spec["figures"]):
            path = FIGURES / rel
            if path.exists():
                col.image(str(path), width="stretch")

st.markdown("---")
st.subheader("Validation approach")

st.markdown("""
| Model | Split | Why |
|---|---|---|
| Churn | Chronological train / val / test (60/20/20) | Target is a future event. Threshold tuned on validation so test stays untouched. |
| Forecast | 5-fold rolling-origin, expanding window | One split would be a single lucky estimate. |
| Recommender | Co-purchase pairs split by time | Pairs are stored in both directions, so a random split hands the model the answer. |
| Low review score | Chronological train / val / test (60/20/20) | Same as churn; selection on validation, reporting on test. |
| Sentiment | Chronological by review date | Vocabulary and complaint mix drift over time. |

**No random `train_test_split` anywhere in this project.** Every target either
is a future event or sits in a time-ordered stream.
""")

action_box("How to use these models responsibly", [
    "<b>Trust the forecast at 30 days</b> and the sentiment classifier. Both beat "
    "their baselines by a clear margin on honest splits.",
    "<b>Use the recommender for cold-start slots</b> — product pages and "
    "post-purchase email — not for personalised feeds.",
    "<b>Use churn only for narrow targeting</b> (top 1%) where lift reaches 2.7×. "
    "Do not present it as identifying at-risk customers at scale.",
    "<b>Use the low-review model operationally.</b> The top 1% of orders are "
    "flagged with 70.8% precision — enough for a pre-emptive service contact "
    "before the review is written.",
    "<b>Never quote the sentiment model as satisfaction.</b> It measures complaint "
    "themes on a corpus biased 2× toward negatives.",
])

caveat("The churn model is weak, and it was reported as weak rather than tuned "
       "until a favourable number appeared. A churn model with 1.32× lift "
       "described accurately is more useful than one with 92% accuracy described "
       "misleadingly. The low-review model is the control: same pipeline, same "
       "code, 3.19× lift — so the difference is the data, not the method.")