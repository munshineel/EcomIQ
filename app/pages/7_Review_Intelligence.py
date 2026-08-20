"""Page 7 - Review Intelligence. What is breaking, and who owns it?"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    BAD, GOOD, PALETTE, WARN, action_box, apply_filters, caveat, kpi_row,
    load_aspects, load_orders, load_sentiment, page_setup, pct,
    require_artifacts, sidebar_filters,
)

page_setup("Review Intelligence",
           "Complaint themes from 40,977 Portuguese reviews — what to fix and who owns it")

require_artifacts("dashboard orders")

orders = load_orders()
sentiment = load_sentiment()
aspects = load_aspects()

filters = sidebar_filters(orders)
df = apply_filters(orders, filters)

joined = df.merge(sentiment, on="order_id", how="inner", suffixes=("", "_s"))
if joined.empty:
    st.warning("No reviewed orders match the current filters.")
    st.stop()

ASPECT_COLS = [c for c in joined.columns if c.startswith("aspect_")]
OWNERS = {
    "aspect_delivery": "Logistics", "aspect_seller_service": "Seller ops",
    "aspect_product_quality": "Merchandising", "aspect_completeness": "Warehouse",
    "aspect_expectation_match": "Content / listings", "aspect_packaging": "Warehouse",
    "aspect_price_value": "Pricing",
}

neg_rate = (joined["review_score"] <= 2).mean()
worst = max(ASPECT_COLS,
            key=lambda c: joined.loc[joined[c], "review_score"].le(2).mean()
            if joined[c].sum() > 30 else 0)

kpi_row([
    ("Reviews with text", f"{len(joined):,}", None),
    ("Mean score", f"{joined['review_score'].mean():.2f} ★", None),
    ("Negative (1–2★)", pct(neg_rate), None),
    ("Worst theme", worst.replace("aspect_", ""),
     f"{joined.loc[joined[worst], 'review_score'].mean():.2f}★"),
])

st.markdown("---")
st.subheader("Complaint themes: how often raised versus how damaging")

rows = []
for col in ASPECT_COLS:
    m = joined[col]
    if m.sum() < 20:
        continue
    sub = joined[m]
    rows.append({
        "Theme": col.replace("aspect_", ""),
        "Mentions": int(m.sum()),
        "Share of reviews": m.mean(),
        "Mean score": sub["review_score"].mean(),
        "% negative": sub["review_score"].le(2).mean(),
        "Owner": OWNERS.get(col, "—"),
    })
theme = pd.DataFrame(rows).sort_values("% negative", ascending=False)

fig = go.Figure(go.Scatter(
    x=theme["Share of reviews"] * 100, y=theme["Mean score"],
    mode="markers+text", text=theme["Theme"], textposition="top center",
    marker=dict(size=np.sqrt(theme["Mentions"]) * 1.6,
                color=theme["% negative"] * 100, colorscale="Reds",
                showscale=True, colorbar=dict(title="% 1–2★"),
                line=dict(width=1, color="#94A3B8")),
    hovertemplate="<b>%{text}</b><br>%{x:.1f}% of reviews<br>%{y:.2f}★<extra></extra>",
))
fig.add_hline(y=joined["review_score"].mean(), line_dash="dash", line_color="#94A3B8",
              annotation_text="corpus average")
fig.update_layout(height=430, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
                  xaxis_title="% of reviews mentioning the theme",
                  yaxis_title="Mean review score")
fig.update_xaxes(gridcolor="#E2E8F0"); fig.update_yaxes(gridcolor="#E2E8F0")
st.plotly_chart(fig, width="stretch")
st.caption("Bottom-left = rare but devastating. Bottom-right = common and "
           "damaging. Bubble size is mention volume.")

st.markdown("---")
c1, c2 = st.columns([1.15, 1])

with c1:
    st.subheader("Theme × category")
    top_cats = joined["category"].value_counts().head(10).index
    matrix = pd.DataFrame({
        col.replace("aspect_", ""): [
            joined.loc[joined[col] & (joined["category"] == c), "review_score"].le(2).mean()
            for c in top_cats
        ] for col in ASPECT_COLS
    }, index=top_cats)
    fig2 = go.Figure(go.Heatmap(z=matrix.to_numpy() * 100, x=matrix.columns,
                                y=matrix.index, colorscale="Reds",
                                colorbar=dict(title="% 1–2★"),
                                hovertemplate="%{y} · %{x}<br>%{z:.0f}% negative<extra></extra>"))
    fig2.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, width="stretch")

with c2:
    st.subheader("Theme detail")
    disp = theme.copy()
    disp["Share of reviews"] = disp["Share of reviews"].map(lambda v: pct(v))
    disp["% negative"] = disp["% negative"].map(lambda v: pct(v))
    disp["Mean score"] = disp["Mean score"].round(2)
    st.dataframe(disp, hide_index=True, width="stretch", height=400)

st.markdown("---")
st.subheader("Read the actual comments")

pick = st.selectbox(
    "Theme", theme["Theme"].tolist(),
    help="Themes were derived from the corpus, not invented: the 60 most "
         "frequent tokens were ranked, grouped, then checked for coverage and "
         "score separation.")
col = f"aspect_{pick}"
only_neg = st.checkbox(
    "Negative reviews only (1–2★)", value=True,
    help="Positive reviews mentioning a theme are usually praise for it, which "
         "is less actionable than the complaints.")

sample = joined[joined[col]]
if only_neg:
    sample = sample[sample["review_score"] <= 2]

if "review_comment_message" in sample.columns and len(sample):
    for _, r in sample.head(8).iterrows():
        st.markdown(f"**{int(r['review_score'])}★** · {r['category']} · "
                    f"{r['customer_state']} — {str(r['review_comment_message'])[:220]}")
elif not len(sample):
    st.info("No comments match this theme under the current filters.")
else:
    st.info("Comment text missing from review_sentiment.csv. Re-run notebook 07 "
            "to include `review_comment_message` in the export.")

action_box("Recommended actions", [
    f"<b>{OWNERS.get(worst, '—')}:</b> '{worst.replace('aspect_', '')}' is the most "
    f"damaging theme at {joined.loc[joined[worst], 'review_score'].mean():.2f}★. "
    "Partial deliveries hurt more than late ones and are invisible in delivery KPIs.",
    "<b>Logistics:</b> delivery is mentioned in 55% of all comments — the single "
    "largest theme by volume. Manage against the promised date, not average speed.",
    "<b>Content / listings:</b> expectation-mismatch complaints point at listing "
    "photos and descriptions, which is a cheap fix relative to logistics.",
])

caveat("This measures COMPLAINT THEMES, not customer satisfaction. 76.6% of "
       "1-star reviewers write a comment versus 35.9% of 5-star, so the text "
       "corpus over-represents negatives by roughly 2×. Low scores are 14.7% of "
       "all reviews but 26.6% of reviews with text.")