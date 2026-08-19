"""Page 3 - Customer Segmentation. Which groups do we treat differently?"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    BAD, GOOD, PALETTE, WARN, action_box, brl, caveat, kpi_row,
    load_segment_profiles, load_segments, page_setup, pct,
)

page_setup("Customer Segmentation", "Six behavioural segments from K-Means (K=6)")

segments = load_segments()
profiles = load_segment_profiles()

st.sidebar.markdown("### Filters")
all_segs = sorted(segments["segment"].dropna().unique())
chosen = st.sidebar.multiselect(
    "Segments", all_segs, default=all_segs,
    help="Six segments from K-Means on 10 behavioural features. Deselect to "
         "compare a subset — shares recompute against the visible customers.")
states = sorted(segments["customer_state"].dropna().unique())
sel_states = st.sidebar.multiselect("State", states, placeholder="All states")

df = segments[segments["segment"].isin(chosen)]
if sel_states:
    df = df[df["customer_state"].isin(sel_states)]
if df.empty:
    st.warning("No customers match the current filters.")
    st.stop()

summary = df.groupby("segment").agg(
    customers=("customer_unique_id", "size"),
    revenue=("total_revenue", "sum"),
    median_revenue=("total_revenue", "median"),
    mean_orders=("n_orders", "mean"),
).sort_values("revenue", ascending=False)
summary["pct_customers"] = summary["customers"] / summary["customers"].sum()
summary["pct_revenue"] = summary["revenue"] / summary["revenue"].sum()

top_value = summary.index[0]
at_risk_segs = ["Delivery failures (at risk)", "Never delivered"]
at_risk_rev = summary.loc[summary.index.isin(at_risk_segs), "revenue"].sum()

kpi_row([
    ("Segments shown", f"{len(summary)}", None),
    ("Largest segment", f"{summary['pct_customers'].idxmax()[:22]}",
     pct(summary["pct_customers"].max())),
    ("Highest value", f"{top_value[:22]}", pct(summary.loc[top_value, 'pct_revenue'])),
    ("Revenue at risk", brl(at_risk_rev), None),
])

st.markdown("---")
st.subheader("Segment size versus value")

order = summary.sort_values("pct_revenue")
fig = go.Figure()
fig.add_trace(go.Bar(y=order.index, x=-order["pct_customers"] * 100, orientation="h",
                     name="% of customers", marker_color="#94A3B8",
                     hovertemplate="%{y}<br>%{customdata:.1f}% of customers<extra></extra>",
                     customdata=order["pct_customers"] * 100))
fig.add_trace(go.Bar(y=order.index, x=order["pct_revenue"] * 100, orientation="h",
                     name="% of revenue", marker_color=PALETTE[0],
                     hovertemplate="%{y}<br>%{x:.1f}% of revenue<extra></extra>"))
fig.update_layout(barmode="relative", height=380, plot_bgcolor="white",
                  margin=dict(l=0, r=0, t=10, b=0),
                  xaxis_title="← share of customers      share of revenue →",
                  legend=dict(orientation="h", y=1.15, x=0))
fig.update_xaxes(gridcolor="#E2E8F0",
                 tickvals=[-40, -20, 0, 20, 40], ticktext=["40", "20", "0", "20", "40"])
st.plotly_chart(fig, width="stretch")
st.caption("A segment wider on the right than the left earns more than its size "
           "implies. Wider on the left costs more than it returns.")

st.markdown("---")
st.subheader("Segment profile")

pick = st.selectbox(
    "Inspect a segment", order.index[::-1],
    help="Shows the segment's profile and the treatment its numbers justify.")
row = summary.loc[pick]
prof_match = profiles[profiles.index.isin(range(10))]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{int(row['customers']):,}", pct(row["pct_customers"]))
c2.metric("Revenue", brl(row["revenue"]), pct(row["pct_revenue"]))
c3.metric("Median spend", brl(row["median_revenue"]), None)
c4.metric("Mean orders", f"{row['mean_orders']:.2f}", None)

TREATMENTS = {
    "High-value installment buyers": (
        GOOD, "Protect. 18.5% of customers, 45.8% of revenue. They buy big-ticket "
        "items with the lowest freight ratio (0.11) across 7 installments. "
        "Installment availability is load-bearing — changing credit terms is the "
        "highest-risk decision available. Route to priority fulfilment."),
    "Satisfied mainstream": (
        GOOD, "Maintain. Largest group at 45.7%, best satisfaction (4.65★), "
        "9-day delivery. Nothing is broken here. Do not spend on it; protect the "
        "service level that produced it."),
    "Delivery failures (at risk)": (
        BAD, "Fix operationally. 12.3% of customers and 11.9% of revenue, at "
        "21.8-day delivery and 1.71★. These are not cheap customers — they spend "
        "near the median and were failed. Highest-value fixable segment."),
    "Repeat / multi-item buyers": (
        GOOD, "Study. The only genuine repeat group (2.11 orders, 2.57 items) at "
        "3% of customers. Too small for a programme, large enough to learn from — "
        "understand what they have in common."),
    "Low-value, freight-burdened": (
        WARN, "Change the economics, not the marketing. 18.8% of customers but "
        "3.6% of revenue, with freight at 60% of order value. Minimum-basket "
        "incentives or regional freight subsidy. Retention spend here is wasted."),
    "Never delivered": (
        BAD, "Escalate to operations. 1.7% of customers, no delivery date on "
        "record, 1.79★. This is a process failure, not a marketing problem."),
}
colour, text = TREATMENTS.get(pick, (PALETTE[5], "No treatment defined."))
st.markdown(
    f'<div style="border-left:4px solid {colour};background:#F8FAFC;'
    f'padding:0.9rem 1.1rem;border-radius:4px;margin-top:0.6rem;">'
    f'<strong>Recommended treatment</strong><br>{text}</div>',
    unsafe_allow_html=True)

st.markdown("---")
st.subheader("All segments")
disp = summary.copy()
disp["revenue"] = disp["revenue"].map(brl)
disp["median_revenue"] = disp["median_revenue"].map(brl)
disp["pct_customers"] = disp["pct_customers"].map(lambda v: pct(v))
disp["pct_revenue"] = disp["pct_revenue"].map(lambda v: pct(v))
disp.columns = ["Customers", "Revenue", "Median spend", "Mean orders",
                "% customers", "% revenue"]
st.dataframe(disp, width="stretch")

action_box("Recommended actions", [
    "<b>Protect</b> High-value installment buyers — 45.8% of revenue from 18.5% of customers.",
    f"<b>Fix</b> Delivery failures — {brl(summary.loc['Delivery failures (at risk)', 'revenue']) if 'Delivery failures (at risk)' in summary.index else 'n/a'} "
    "of revenue behind a 21.8-day delivery time and 1.71★.",
    "<b>Reprice</b> Low-value freight-burdened — nearly a fifth of customers are "
    "barely profitable because freight is 60% of order value.",
])

caveat("Silhouette score is ~0.21 across K=4 to K=8 — structure is weak and "
       "overlapping, which is expected when 98% of customers place one order. "
       "K=6 was chosen because geometry could not separate 6/7/8 and six groups "
       "map to six distinct actions. Treat segments as useful groupings, not "
       "hard boundaries.")