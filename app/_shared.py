"""Shared loading, styling and components for the PropheticIQ dashboard.

Every page imports from here so that filters, colours and KPI cards are
identical across the app -- which is most of what separates a product from a
set of notebook exports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

# Matches src/viz/style.py so notebook figures and app charts look like one product.
PALETTE = ["#3B6EA5", "#D97706", "#0F766E", "#B91C1C", "#6D28D9", "#64748B"]
INK = "#1F2937"
MUTED = "#64748B"
GOOD, WARN, BAD = "#0F766E", "#D97706", "#B91C1C"

CSS = """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 6rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 600;}
  [data-testid="stMetricLabel"] {font-size: 0.78rem; color: #64748B;
      text-transform: uppercase; letter-spacing: 0.04em;}
  h1 {font-size: 1.7rem !important; font-weight: 600 !important;}
  h2 {font-size: 1.2rem !important; font-weight: 600 !important;
      margin-top: 1.6rem !important;}
  h3 {font-size: 1.0rem !important; font-weight: 600 !important;}
  .action-box {background:#F8FAFC; border-left:4px solid #3B6EA5;
      padding:0.9rem 1.1rem; border-radius:4px; margin-top:0.5rem;}
  .caveat {background:#FEF3C7; border-left:4px solid #D97706;
      padding:0.7rem 1rem; border-radius:4px; font-size:0.87rem;}
  hr {margin: 1.2rem 0;}
</style>
"""


def page_setup(title: str, subtitle: str = "") -> None:
    st.set_page_config(page_title=f"PropheticIQ · {title}", page_icon="📊",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # Brand in the sidebar so it appears on every page and in screenshots,
    # without competing with each page's own title.
    st.sidebar.markdown(
        "<div style='font-size:1.15rem;font-weight:700;color:#3B6EA5;"
        "letter-spacing:-0.01em;'>PropheticIQ</div>"
        "<div style='font-size:0.72rem;color:#64748B;margin-bottom:0.9rem;'>"
        "E-Commerce Intelligence &amp; Decision Platform</div>",
        unsafe_allow_html=True,
    )

    st.title(title)
    if subtitle:
        st.caption(subtitle)


# ---------------------------------------------------------------------------
# cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_orders() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "dashboard_orders.parquet")


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "customer_segments.csv")


@st.cache_data(show_spinner=False)
def load_segment_profiles() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "segment_profiles.csv", index_col=0)


@st.cache_data(show_spinner=False)
def load_daily_revenue() -> pd.Series:
    df = pd.read_csv(PROCESSED / "daily_revenue.csv", index_col=0, parse_dates=[0])
    return df["revenue"]


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "revenue_forecast_90d.csv", index_col=0, parse_dates=[0])


@st.cache_data(show_spinner=False)
def load_sentiment() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "review_sentiment.csv")


@st.cache_data(show_spinner=False)
def load_aspects() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "aspect_summary.csv", index_col=0)


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "product_catalog.parquet")


@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "top_recommendations.parquet")


@st.cache_data(show_spinner=False)
def load_churn_features() -> pd.DataFrame:
    p = PROCESSED / "churn_features.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(
        PROCESSED / "churn_features.csv", parse_dates=["order_purchase_timestamp"])


@st.cache_data(show_spinner=False)
def load_comparison(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / name, index_col=0)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """The single model-access point for the whole UI.

    Layer boundary: pages never open a .joblib file, never see a feature list,
    and never call predict() themselves. They ask the pipeline for a scored
    DataFrame. That keeps the threshold contract in exactly one place --
    src/inference/pipeline.py -- so it cannot drift between training and
    serving.

    cache_resource, not cache_data: a pipeline holds unhashable sklearn objects
    and is shared across all sessions rather than copied per user.
    """
    from src.inference.pipeline import PropheticIQPipeline
    return PropheticIQPipeline()


@st.cache_data(show_spinner="Scoring customers...", ttl=3600)
def score_churn_cached(n_tail: int = 11_410) -> pd.DataFrame:
    """Score the hold-out tail once per session, not once per slider move.

    Without this cache, dragging the contact-budget slider re-ran a Random
    Forest over 11,410 rows on every frame. The scores do not depend on the
    slider -- only the threshold applied to them does -- so scoring is cached
    and thresholding stays cheap and live.

    Only the hold-out tail is scored: the model trained on the earlier 80%, so
    scoring everything would report training-set performance as if it were
    live performance.
    """
    pipe = get_pipeline()
    churn = load_churn_features().sort_values("order_purchase_timestamp")
    holdout = churn.tail(n_tail).copy()
    return pipe.score_churn(holdout)


def require_artifacts(*names: str) -> None:
    """Fail with instructions instead of a traceback.

    A missing artifact is the most likely failure on a fresh clone, and the
    default Streamlit response is a raw stack trace that tells the user
    nothing actionable.
    """
    try:
        health = get_pipeline().health_check()
    except Exception as exc:                       # noqa: BLE001
        st.error("Could not initialise the inference pipeline.")
        st.exception(exc)
        st.stop()

    wanted = health[health["artifact"].isin(names)] if names else health
    missing = wanted[~wanted["present"]]
    if len(missing):
        st.error(f"{len(missing)} required artifact(s) are missing.")
        st.dataframe(missing[["artifact", "regenerate_with"]], hide_index=True,
                     width="stretch")
        st.info("Run the commands above from the project root, then reload "
                "this page.")
        st.stop()


# ---------------------------------------------------------------------------
# global filters
# ---------------------------------------------------------------------------
def sidebar_filters(orders: pd.DataFrame, show_category: bool = True) -> dict:
    """Global filters, persisted in session_state so they survive navigation."""
    st.sidebar.markdown("### Filters")

    dmin = orders["order_purchase_timestamp"].min().date()
    dmax = orders["order_purchase_timestamp"].max().date()
# Default to the last 6 months rather than the full range, so a comparable
# prior period always exists and the period-over-period deltas render.
# Selecting the full range means the prior window falls before the data
# starts, and any delta shown there would be fabricated.
    six_months_ago = max(dmin, (pd.Timestamp(dmax) - pd.DateOffset(months=6)).date())
    default = st.session_state.get("date_range", (six_months_ago, dmax))
    date_range = st.sidebar.date_input(
        "Date range", value=default, min_value=dmin, max_value=dmax,
        help="Applies to every page and persists as you navigate. Limited to "
             "Jan 2017 - Aug 2018; outside that window Olist is too sparse "
             "to analyse.")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        st.session_state["date_range"] = date_range
    else:
        date_range = st.session_state.get("date_range", (dmin, dmax))

    states = sorted(orders["customer_state"].dropna().unique())
    sel_states = st.sidebar.multiselect(
        "Customer state", states, default=st.session_state.get("states", []),
        placeholder="All states",
        help="Brazilian state of the delivery address. SP alone is 42% of "
             "customers.")
    st.session_state["states"] = sel_states

    sel_cats: list[str] = []
    if show_category:
        cats = sorted(orders["category"].dropna().unique())
        sel_cats = st.sidebar.multiselect(
            "Category", cats, default=st.session_state.get("cats", []),
            placeholder="All categories",
            help="Category of the most expensive item in each order, since an "
                 "order can span several categories.")
        st.session_state["cats"] = sel_cats

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: Olist Brazilian E-Commerce, Jan 2017 – Aug 2018. "
        "Revenue excludes freight."
    )
    return {"dates": date_range, "states": sel_states, "categories": sel_cats}


def apply_filters(orders: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = orders
    start, end = pd.Timestamp(f["dates"][0]), pd.Timestamp(f["dates"][1]) + pd.Timedelta(days=1)
    out = out[out["order_purchase_timestamp"].between(start, end)]
    if f["states"]:
        out = out[out["customer_state"].isin(f["states"])]
    if f.get("categories"):
        out = out[out["category"].isin(f["categories"])]
    return out


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------
def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Four KPIs in a row: (label, value, delta or None)."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def action_box(title: str, lines: list[str]) -> None:
    """Recommended-actions block.

    Uses a native bordered container rather than injected HTML. Custom divs
    sit outside Streamlit's layout engine, so they do not reserve height and
    the last one on a page ends up clipped against the viewport edge.
    """
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for line in lines:
            st.markdown(f"- {line}", unsafe_allow_html=True)


def caveat(text: str) -> None:
    """Limitation or warning. st.warning renders inside the layout flow."""
    st.warning(text, icon="⚠️")


def brl(x: float) -> str:
    if abs(x) >= 1e6:
        return f"R$ {x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"R$ {x / 1e3:.0f}k"
    return f"R$ {x:.0f}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"