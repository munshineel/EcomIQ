"""Page 2 - Customer Intelligence. Is targeted retention viable at all?"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    PALETTE, action_box, apply_filters, brl, kpi_row, load_orders, load_segments,
    page_setup, pct, sidebar_filters,
)

page_setup("Customer Intelligence", "Who our customers are, and whether targeting them pays")

orders = load_orders()
filters = sidebar_filters(orders, show_category=False)
df = apply_filters(orders, filters)
fulfilled = df[df["is_fulfilled"] & df["revenue"].notna()]

if fulfilled.empty:
    st.warning("No orders match the current filters.")
    st.stop()

per_customer = fulfilled.groupby("customer_unique_id").agg(
    revenue=("revenue", "sum"), orders=("order_id", "count"),
    state=("customer_state", "first"), review=("review_score", "mean"),
)

# Gini on the sorted cumulative share - O(n log n), unlike the pairwise formula.
rev = np.sort(per_customer["revenue"].to_numpy())
idx = np.arange(1, len(rev) + 1)
gini = float((2 * idx - len(rev) - 1).dot(rev) / (len(rev) * rev.sum()))
cum = np.cumsum(rev) / rev.sum()
pop = idx / len(rev)
top10 = 1 - np.interp(0.90, pop, cum)

repeat_rate = (per_customer["orders"] > 1).mean()

kpi_row([
    ("Customers", f"{len(per_customer):,}", None),
    ("Repeat rate", pct(repeat_rate, 2), None),
    ("Revenue concentration (Gini)", f"{gini:.3f}", None),
    ("Top 10% revenue share", pct(top10), None),
])

st.markdown("---")

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Revenue concentration")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
                             line=dict(color="#94A3B8", dash="dash", width=1)))
    step = max(1, len(pop) // 2000)          # thin for render speed, shape unchanged
    fig.add_trace(go.Scatter(x=pop[::step], y=cum[::step], mode="lines", name="Observed",
                             fill="tonexty", line=dict(color=PALETTE[0], width=2.5)))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="white",
                      xaxis_title="Cumulative share of customers (lowest to highest spend)",
                      yaxis_title="Cumulative share of revenue",
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor="#E2E8F0"); fig.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig, width="stretch")
    st.caption("A Lorenz curve answers 'how unequal' on one axis. A histogram "
               "would only show spread.")

with right:
    st.subheader("Value by state")
    by_state = fulfilled.groupby("customer_state").agg(
        customers=("customer_unique_id", "nunique"),
        median_order=("revenue", "median"),
        freight_ratio=("freight_ratio", "median"),
        delivery_days=("delivery_days", "median"),
    ).sort_values("customers", ascending=False).head(12)
    by_state.columns = ["Customers", "Median order (R$)", "Freight ratio", "Delivery days"]
    st.dataframe(by_state.round(2), width="stretch", height=380)
    st.caption("Sort any column. Geography here is a cost story, not a revenue one.")

st.markdown("---")
st.subheader("Where the money sits")

deciles = pd.qcut(per_customer["revenue"], 10, labels=False, duplicates="drop")
dec = per_customer.groupby(deciles).agg(
    customers=("revenue", "size"), revenue=("revenue", "sum"))
dec["share"] = dec["revenue"] / dec["revenue"].sum()

fig = go.Figure(go.Bar(x=[f"D{i+1}" for i in dec.index], y=dec["share"] * 100,
                       marker_color=PALETTE[0]))
fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
                  xaxis_title="Customer spend decile (D10 = highest)",
                  yaxis_title="% of total revenue")
fig.update_yaxes(gridcolor="#E2E8F0")
st.plotly_chart(fig, width="stretch")

viable = top10 > 0.30
action_box("Recommended actions", [
    f"<b>Concentration is real:</b> Gini {gini:.3f} and the top decile holds "
    f"{top10:.1%} of revenue — despite a {repeat_rate:.2%} repeat rate. "
    f"High-value customers are identifiable from a <em>single</em> purchase.",
    ("<b>Targeted treatment is viable.</b> Route the top decile to priority "
     "fulfilment and freight subsidy; the concentration justifies differentiated "
     "service." if viable else
     "<b>Targeted treatment is not viable.</b> Revenue is too evenly spread — "
     "spend on acquisition and first-order experience instead."),
    "<b>Do not build a loyalty programme around repeat purchase.</b> At under 2% "
    "repeat rate there is no base. Value must be captured on order one.",
])