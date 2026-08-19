"""Page 1 - Executive Overview. Entry point: streamlit run app/Home.py"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    BAD, GOOD, PALETTE, WARN, action_box, apply_filters, brl, caveat, kpi_row,
    load_aspects, load_daily_revenue, load_forecast, load_orders, load_segments,
    page_setup, pct, require_artifacts, sidebar_filters,
)

page_setup("Executive Overview", "Marketplace performance and the risks behind it")

require_artifacts("dashboard orders", "segments", "forecast CSV")

orders = load_orders()
filters = sidebar_filters(orders)
df = apply_filters(orders, filters)
fulfilled = df[df["is_fulfilled"]]

if fulfilled.empty:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------- KPIs
# Period-over-period: compare the selected window against the equally long
# window immediately before it. A KPI without a comparison is trivia.
start, end = pd.Timestamp(filters["dates"][0]), pd.Timestamp(filters["dates"][1])
span = end - start
prev = orders[
    orders["is_fulfilled"]
    & orders["order_purchase_timestamp"].between(start - span, start)
]
if filters["states"]:
    prev = prev[prev["customer_state"].isin(filters["states"])]
if filters["categories"]:
    prev = prev[prev["category"].isin(filters["categories"])]


def delta(cur: float, before: float) -> str | None:
    if not before:
        return None
    return f"{(cur / before - 1) * 100:+.1f}%"


revenue = fulfilled["revenue"].sum()
n_orders = len(fulfilled)
aov = fulfilled["revenue"].mean()
review = fulfilled["review_score"].mean()

kpi_row([
    ("Revenue", brl(revenue), delta(revenue, prev["revenue"].sum())),
    ("Orders", f"{n_orders:,}", delta(n_orders, len(prev))),
    ("Avg order value", brl(aov), delta(aov, prev["revenue"].mean())),
    ("Mean review", f"{review:.2f} ★", delta(review, prev["review_score"].mean())),
])

st.markdown("---")

# ---------------------------------------------------------------- primary chart
st.subheader("Revenue trend and 30-day forecast")

metric = st.radio(
    "Show", ["Revenue", "Orders"], horizontal=True, label_visibility="collapsed",
    help="Revenue excludes freight, which is a cost pass-through rather than "
         "product revenue.")

monthly = (
    fulfilled.set_index("order_purchase_timestamp")
    .resample("MS")
    .agg(revenue=("revenue", "sum"), orders=("order_id", "count"))
)
series = monthly["revenue"] if metric == "Revenue" else monthly["orders"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers",
                         name="Actual", line=dict(color=PALETTE[0], width=2.5)))

# Forecast is only meaningful with no filters applied - it was fitted on the
# whole marketplace, so showing it beside a filtered subset would mislead.
unfiltered = not filters["states"] and not filters["categories"]
if metric == "Revenue" and unfiltered:
    fc = load_forecast().iloc[:30]
    fc_monthly = fc["forecast"].resample("MS").sum()
    fc_monthly = fc_monthly[fc_monthly > 0]
    fig.add_trace(go.Scatter(x=fc_monthly.index, y=fc_monthly.values,
                             mode="lines+markers", name="Forecast (SARIMA)",
                             line=dict(color=PALETTE[1], width=2.5, dash="dash")))

fig.update_layout(
    height=380, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
    plot_bgcolor="white", yaxis_title=metric,
    legend=dict(orientation="h", y=1.12, x=0),
)
fig.update_yaxes(gridcolor="#E2E8F0")
st.plotly_chart(fig, width="stretch")

if not unfiltered:
    st.caption("Forecast hidden: it was fitted on total marketplace revenue, "
               "so it is not comparable to a filtered subset.")

# ---------------------------------------------------------------- risk register
st.markdown("---")
st.subheader("Top risks, ranked by revenue at stake")

segments = load_segments()
aspects = load_aspects()

late = fulfilled[fulfilled["is_late"] == True]  # noqa: E712
late_rev = late["revenue"].sum()
late_rate = len(late) / len(fulfilled)

at_risk = segments[segments["segment"] == "Delivery failures (at risk)"]
never = segments[segments["segment"] == "Never delivered"]
freight_burdened = fulfilled[fulfilled["freight_ratio"] > 0.5]

risks = pd.DataFrame([
    {
        "Risk": "Late deliveries",
        "Revenue at stake": late_rev,
        "Scale": f"{late_rate:.1%} of orders",
        "Impact": f"{late['review_score'].mean():.2f}★ vs "
                  f"{fulfilled.loc[~fulfilled['is_late'].fillna(False), 'review_score'].mean():.2f}★ on time",
        "Owner": "Logistics",
    },
    {
        "Risk": "Incomplete orders (missing items)",
        "Revenue at stake": float(aspects.loc["completeness", "mentions"]) * aov,
        "Scale": f"{aspects.loc['completeness', 'share_of_comments']:.1%} of reviews",
        "Impact": f"{aspects.loc['completeness', 'mean_score']:.2f}★ — worst aspect",
        "Owner": "Warehouse",
    },
    {
        "Risk": "Freight-burdened orders",
        "Revenue at stake": float(freight_burdened["revenue"].sum()),
        "Scale": f"{len(freight_burdened) / len(fulfilled):.1%} of orders",
        "Impact": "Freight exceeds 50% of product value",
        "Owner": "Pricing",
    },
]).sort_values("Revenue at stake", ascending=False)

risks_display = risks.copy()
risks_display["Revenue at stake"] = risks_display["Revenue at stake"].map(brl)
st.dataframe(risks_display, hide_index=True, width="stretch")

# ---------------------------------------------------------------- actions
worst_aspect = aspects["pct_negative"].idxmax()
action_box("Recommended actions", [
    f"<b>Logistics:</b> {late_rate:.1%} of orders arrive after the promised date, "
    f"costing {brl(late_rev)} in affected revenue. Reviews collapse from 4.15★ to "
    f"2.32★ once an order is 3+ days late — the promise date, not average speed, "
    f"is the metric to manage.",
    f"<b>Warehouse:</b> '{worst_aspect}' is the most damaging complaint theme at "
    f"{aspects.loc[worst_aspect, 'mean_score']:.2f}★ mean score. Partial deliveries "
    f"hurt more than late ones and are invisible in delivery KPIs.",
    f"<b>Retention:</b> repeat rate is 1.92%. Do not fund a broad loyalty "
    f"programme — see Churn Prediction for the top-1% targeting case instead.",
])

caveat("Review text over-represents complaints: 76.6% of 1-star reviewers write "
       "a comment versus 35.9% of 5-star. Treat review themes as complaint "
       "signal, not overall satisfaction.")