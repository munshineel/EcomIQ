"""Page 4 - Churn Prediction. Who do we contact this week, and does it pay?

Layer discipline: this file contains NO model loading, NO feature list and NO
predict() call. It asks `score_churn_cached()` for a scored frame and spends the
rest of its lines on presentation. Swapping Random Forest for a neural network
would not change a line in here.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    PALETTE, action_box, brl, caveat, get_pipeline, kpi_row, load_comparison,
    page_setup, pct, require_artifacts, score_churn_cached,
)

page_setup("Churn Prediction",
           "Which first-time buyers return within 180 days — and who to contact")

# Fails with regeneration instructions rather than a stack trace.
require_artifacts("churn model", "dashboard orders")

meta = get_pipeline().churn_metadata

with st.spinner("Loading scored customers..."):
    holdout = score_churn_cached()

base_rate = holdout["target_repeat"].mean()
total_returners = int(holdout["target_repeat"].sum())

# --------------------------------------------------------------- controls
st.sidebar.markdown("### Campaign settings")
budget_pct = st.sidebar.slider(
    "Contact top % of customers", 1, 20, 5, 1,
    help="How many customers the campaign can afford to reach. The model ranks "
         "everyone by predicted return probability and this cuts the list. "
         "Narrower is always more efficient — the question is whether it reaches "
         "enough people to matter.",
)
offer_cost = st.sidebar.number_input(
    "Cost per contact (R$)", min_value=0.10, max_value=200.0, value=5.0, step=0.50,
    help="Fully loaded cost of reaching one customer: discount value, email or "
         "SMS delivery, and handling.",
)
margin = st.sidebar.number_input(
    "Margin per returning customer (R$)", min_value=1.0, max_value=1000.0,
    value=40.0, step=5.0,
    help="Contribution margin if a customer returns and orders again. Median "
         "order revenue is about R$ 87, so R$ 30–50 is a reasonable starting "
         "assumption for a commission model.",
)

# Input validation: if the offer costs more than the customer is worth, no model
# can rescue the economics. Say so rather than rendering a negative-ROI table.
if offer_cost >= margin:
    st.warning(
        f"Cost per contact (R$ {offer_cost:.2f}) is at or above the margin per "
        f"returning customer (R$ {margin:.2f}). Even perfect precision loses "
        f"money at these numbers. Lower the offer cost or revisit the margin.",
        icon="⚠️",
    )

# ------------------------------------------------------------ thresholding
# Cheap: reuses the cached scores, only the cutoff changes.
threshold = float(np.quantile(holdout["repeat_proba"], 1 - budget_pct / 100))
flagged = holdout[holdout["repeat_proba"] >= threshold]

found = int(flagged["target_repeat"].sum())
precision = found / len(flagged) if len(flagged) else 0.0
recall = found / total_returners if total_returners else 0.0
lift = precision / base_rate if base_rate else 0.0

campaign_cost = len(flagged) * offer_cost
campaign_return = found * margin
roi = (campaign_return - campaign_cost) / campaign_cost if campaign_cost else 0.0

kpi_row([
    ("Base repeat rate", pct(base_rate, 2), None),
    ("Customers flagged", f"{len(flagged):,}", f"top {budget_pct}%"),
    ("Returners found", f"{found}", f"{recall:.1%} of all"),
    ("Lift vs random", f"{lift:.2f}×", "viable" if lift >= 1.5 else "marginal"),
])

st.markdown("---")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Efficiency across contact budgets")
    st.caption("A wider net catches more returners but wastes more contacts. "
               "Below about 1.5× lift a campaign is not worth running.")

    rows = []
    for p in range(1, 21):
        cut = np.quantile(holdout["repeat_proba"], 1 - p / 100)
        sub = holdout.loc[holdout["repeat_proba"] >= cut, "target_repeat"]
        rows.append({"budget": p,
                     "lift": (sub.mean() / base_rate) if len(sub) else 0.0,
                     "recall": sub.sum() / total_returners if total_returners else 0.0})
    sweep = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["budget"], y=sweep["lift"], mode="lines+markers",
                             name="Lift vs random",
                             line=dict(color=PALETTE[0], width=2.5)))
    fig.add_trace(go.Scatter(x=sweep["budget"], y=sweep["recall"] * 10, mode="lines",
                             name="Recall (×10 for scale)",
                             line=dict(color=PALETTE[2], width=2, dash="dot")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#94A3B8",
                  annotation_text="random targeting")
    fig.add_hline(y=1.5, line_dash="dot", line_color=PALETTE[1],
                  annotation_text="viability floor")
    fig.add_vline(x=budget_pct, line_color=PALETTE[3], line_width=2)
    fig.update_layout(height=360, plot_bgcolor="white",
                      margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Contact top % of customers", yaxis_title="Lift",
                      legend=dict(orientation="h", y=1.14, x=0))
    fig.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Campaign economics")
    econ = pd.DataFrame({
        "Metric": ["Customers contacted", "Cost per contact", "Total cost",
                   "Returners found", "Margin each", "Gross return", "Net", "ROI"],
        "Value": [f"{len(flagged):,}", brl(offer_cost), brl(campaign_cost),
                  f"{found}", brl(margin), brl(campaign_return),
                  brl(campaign_return - campaign_cost), f"{roi:+.0%}"],
    })
    st.dataframe(econ, hide_index=True, width="stretch", height=330)

    if roi > 0:
        st.success(f"Profitable at these assumptions: {roi:+.0%} ROI on "
                   f"{brl(campaign_cost)} of spend.")
    else:
        st.error(f"Loses money: {roi:+.0%} ROI. Narrow the budget, cut the offer "
                 f"cost, or do not run the campaign.")

st.markdown("---")
st.subheader("Flagged customers")
st.caption("Highest predicted return probability first. `Actually returned` is "
           "the observed outcome — available here because this is held-out "
           "historical data, and absent in live use.")

show = flagged.nlargest(200, "repeat_proba")[[
    "customer_unique_id", "repeat_proba", "revenue", "review_score",
    "delivery_days", "days_late", "customer_state", "primary_category",
    "target_repeat",
]].copy()
show["repeat_proba"] = show["repeat_proba"].round(4)
show.columns = ["Customer", "Return probability", "First order (R$)", "Review",
                "Delivery days", "Days late", "State", "Category",
                "Actually returned"]
st.dataframe(show, hide_index=True, width="stretch", height=330)

st.download_button(
    "Download targeting list (CSV)",
    flagged[["customer_unique_id", "repeat_proba"]].to_csv(index=False),
    file_name=f"churn_targets_top{budget_pct}pct.csv", mime="text/csv",
    help="Customer IDs and scores for the current budget, ready to hand to a "
         "campaign tool.",
)

with st.expander("Model details and comparison"):
    st.caption(f"Deployed: **{meta['model']}** · {meta['n_features']} features · "
               f"validation-tuned threshold {meta['threshold']:.4f}")
    try:
        st.dataframe(
            load_comparison("churn_model_comparison.csv")[
                ["pr_auc", "pr_auc_lift", "roc_auc", "precision", "recall", "f1"]
            ].round(4), width="stretch")
    except (FileNotFoundError, KeyError):
        st.info("Comparison table missing or malformed. Re-run "
                "notebooks/04_churn_model.py.")

action_box("Recommended actions", [
    (f"<b>Contact the top {budget_pct}%</b> ({len(flagged):,} customers) for "
     f"{lift:.2f}× lift over random targeting."
     if lift >= 1.5 else
     f"<b>Do not run a broad campaign at {budget_pct}%</b> — lift is only "
     f"{lift:.2f}×. Narrow to the top 1–5%, where lift reaches about 2×."),
    "<b>Fix delivery instead.</b> Missed returners averaged 14.5 days to "
    "delivery versus 9.8 for those the model caught. Operational improvement "
    "moves this number more than targeting does.",
    "<b>Do not fund a broad loyalty programme.</b> With about 1,000 positive "
    "examples the model ranks weakly; narrow, cheap targeting is the honest use.",
])

caveat("PR-AUC lift is 1.32× and ROC-AUC 0.55 — the logistic baseline BEAT "
       "Random Forest and XGBoost on validation, meaning there is no "
       "exploitable non-linear structure. This is a weak model reported "
       "honestly. Olist records no browsing, wishlist or price-comparison "
       "signal, so behaviour from a single order is only weakly predictive. "
       "The low-review model on Model Performance uses the same pipeline and "
       "reaches 3.19× lift — the difference is the data, not the method.")