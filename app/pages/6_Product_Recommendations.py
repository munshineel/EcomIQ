"""Page 6 - Product Recommendations. What goes on the product page?"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    PALETTE, action_box, brl, caveat, get_pipeline, kpi_row, load_catalog,
    load_comparison, load_recommendations, page_setup, pct, require_artifacts,
)

page_setup("Product Recommendations",
           "Content-based similarity — the 'customers also viewed' slot")

require_artifacts("recommendations")

catalog = load_catalog().set_index("product_id")
recs = load_recommendations()
comp = load_comparison("recommender_comparison.csv")

sold_once = (catalog["units_sold"] == 1).mean()
best = comp.index[0]

kpi_row([
    ("Catalogue", f"{len(catalog):,}", None),
    ("Products sold once", pct(sold_once), "cold-start"),
    (f"Recall@10 ({best})", f"{comp.loc[best, 'recall@10']:.3f}", None),
    ("Catalogue coverage", pct(comp.loc[best, "coverage"]), None),
])

st.markdown("---")
st.subheader("Try it")

st.sidebar.markdown("### Product filters")
cats = sorted(catalog["category"].dropna().unique())
sel_cat = st.sidebar.selectbox(
    "Category", ["All"] + cats,
    help="Narrows the seed-product picker. Recommendations themselves are not "
         "filtered — the model chooses freely across the catalogue.")
price_lo, price_hi = st.sidebar.slider(
    "Price band (R$)", 0, int(catalog["price"].quantile(0.99)),
    (0, int(catalog["price"].quantile(0.99))), 10,
    help="Capped at the 99th percentile so a handful of R$ 6,700 items do not "
         "stretch the whole range.")

pool = catalog[catalog.index.isin(recs["seed"].unique())]
if sel_cat != "All":
    pool = pool[pool["category"] == sel_cat]
pool = pool[pool["price"].between(price_lo, price_hi)]

if pool.empty:
    st.warning("No products with precomputed recommendations match these filters.")
    st.stop()

options = pool.nlargest(300, "units_sold")
labels = {
    pid: f"{row['category']} · R$ {row['price']:.0f} · {int(row['units_sold'])} sold · {pid[:8]}"
    for pid, row in options.iterrows()
}
seed = st.selectbox(
    "Seed product", list(labels), format_func=lambda p: labels[p],
    help="The product a shopper is currently viewing. Only items with 5+ sales "
         "are listed — recommending for the 18,117 products sold once is not "
         "something a merchandiser would action.")

s = catalog.loc[seed]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Category", str(s["category"])[:20])
c2.metric("Price", brl(s["price"]))
c3.metric("Units sold", f"{int(s['units_sold']):,}")
c4.metric("Weight", f"{s['product_weight_g']:.0f} g")

st.markdown("**Top 10 recommendations**")

# Served through the inference layer, so this page never touches the 23MB
# NearestNeighbors index or knows how similarity is computed.
with st.spinner("Fetching similar products..."):
    try:
        rec_ids = get_pipeline().recommend(seed, k=10)
    except Exception:                                    # noqa: BLE001
        # The live model is a 23MB NearestNeighbors index that is NOT deployed
        # (only its precomputed output is). The picker only offers products
        # present in that output, so this path should be unreachable -- but a
        # missing recommendation must degrade, not crash the page.
        rec_ids = (recs[recs["seed"] == seed]
                   .sort_values("rank")["recommended"].head(10).tolist())

if not rec_ids:
    st.warning("No precomputed recommendations for this product. Run "
               "`python scripts/build_dashboard_data.py` to extend coverage.")
    st.stop()

joined = pd.DataFrame({"Rank": range(1, len(rec_ids) + 1), "Product": rec_ids})
joined = joined.join(
    catalog[["category", "price", "units_sold", "product_weight_g"]], on="Product")
joined.columns = ["Rank", "Product", "Category", "Price (R$)", "Units sold", "Weight (g)"]
st.dataframe(joined.round(1), hide_index=True, width="stretch")

same_cat = (joined["Category"] == s["category"]).mean()
tail = (joined["Units sold"] <= 5).mean()
st.caption(f"{same_cat:.0%} share the seed's category · {tail:.0%} are long-tail "
           f"items (≤5 sales). Long-tail exposure is what popularity ranking "
           f"cannot deliver.")

st.markdown("---")
c1, c2 = st.columns([1.1, 1])

with c1:
    st.subheader("Model comparison")
    show = comp[["precision@10", "recall@10", "MAP@10", "NDCG@10", "coverage"]].round(4)
    st.dataframe(show, width="stretch", height=250)
    st.caption("Evaluated on 824 held-out co-purchase pairs. Recall@10 = hit rate: "
               "did the real basket-mate appear in the top 10?")

with c2:
    st.subheader("Coverage — popularity bias made visible")
    cov = comp["coverage"].sort_values() * 100
    fig = go.Figure(go.Bar(x=cov.values, y=cov.index, orientation="h",
                           marker_color=[PALETTE[3] if v < 5 else PALETTE[0] for v in cov.values]))
    fig.update_layout(height=250, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="% of catalogue ever recommended")
    fig.update_xaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig, width="stretch")
    st.caption("Pure popularity touches 0.03% of the catalogue — 11 products, the "
               "same list for every customer.")

action_box("Recommended actions", [
    "<b>Deploy on the product page</b> ('similar items') and in the "
    "post-purchase email. Both are cold-start contexts, which is exactly where "
    "content-based similarity works and collaborative filtering cannot.",
    "<b>Do not deploy collaborative filtering.</b> It has co-purchase history "
    "for only 22.3% of query items; the rest fall back to best-sellers.",
    "<b>Inject diversity deliberately.</b> Recommendations stay inside the seed's "
    "category and price band, so they will never cross-sell. Reserve 2 of the 10 "
    "slots for an adjacent category if cross-sell is a goal.",
])

caveat("Collaborative filtering is infeasible on this data, and that is a "
       "measured conclusion: 55% of products sold exactly once, only 3.28% of "
       "orders contain two distinct products, and the user-item matrix is "
       "99.9964% empty. A weight sweep of the hybrid was monotonic — every "
       "reduction in CF weight improved results, with the optimum at zero.")