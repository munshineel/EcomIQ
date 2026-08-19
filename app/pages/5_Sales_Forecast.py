"""Page 5 - Sales Forecast. How much do we stock and staff?"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    PALETTE, action_box, brl, caveat, kpi_row, load_comparison,
    load_daily_revenue, load_forecast, page_setup, require_artifacts,
)

page_setup("Sales Forecast", "SARIMA(1,1,1)(1,1,1,7) on daily marketplace revenue")

require_artifacts("forecast CSV")

series = load_daily_revenue()
forecast = load_forecast()
backtest = load_comparison("forecast_backtest.csv")

st.sidebar.markdown("### Forecast settings")
horizon = st.sidebar.select_slider(
    "Horizon (days)", [30, 60, 90], value=30,
    help="Use 30 days for decisions. Intervals widen sharply beyond that — the "
         "90-day band crosses zero, because 20 months of history cannot pin "
         "down a quarter.")
granularity = st.sidebar.radio(
    "Granularity", ["Daily", "Weekly"], horizontal=True,
    help="Daily preserves the weekly rhythm, which is what staffing needs. "
         "Weekly is easier to read for revenue planning.")
show_ci = st.sidebar.checkbox(
    "Show 80% confidence interval", value=True,
    help="Stock against the upper bound and commit cash against the lower one. "
         "Planning to the point forecast means stocking out half the time.")
lookback = st.sidebar.slider(
    "History shown (days)", 60, 365, 120, 30,
    help="How much actual history to draw beside the forecast.")

fc = forecast.iloc[:horizon]
sarima_mae = float(backtest.loc["sarima", "MAE"])
naive_mae = float(backtest.loc["seasonal_naive", "MAE"])

kpi_row([
    (f"Next {horizon}d revenue", brl(fc["forecast"].sum()), None),
    ("Lower bound (80%)", brl(fc["lower_80"].sum()), None),
    ("Upper bound (80%)", brl(fc["upper_80"].sum()), None),
    ("Backtest MAE", brl(sarima_mae), f"{(sarima_mae / naive_mae - 1) * 100:+.0f}% vs naive"),
])

st.markdown("---")
st.subheader(f"{horizon}-day forecast")

hist = series.iloc[-lookback:]
if granularity == "Weekly":
    hist = hist.resample("W").sum()
    plot_fc = fc.resample("W").sum()
else:
    plot_fc = fc

fig = go.Figure()
if show_ci:
    fig.add_trace(go.Scatter(x=list(plot_fc.index) + list(plot_fc.index[::-1]),
                            y=list(plot_fc["upper_80"]) + list(plot_fc["lower_80"][::-1]),
                            fill="toself", fillcolor="rgba(59,110,165,0.16)",
                            line=dict(color="rgba(0,0,0,0)"), name="80% interval",
                            hoverinfo="skip"))
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Actual",
                         line=dict(color="#94A3B8", width=1.6)))
fig.add_trace(go.Scatter(x=plot_fc.index, y=plot_fc["forecast"], mode="lines",
                         name="Forecast", line=dict(color=PALETTE[0], width=2.6)))
fig.add_vline(x=series.index[-1], line_dash="dash", line_color=PALETTE[3])
fig.update_layout(height=420, plot_bgcolor="white", hovermode="x unified",
                  margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Revenue (R$)",
                  legend=dict(orientation="h", y=1.12, x=0))
fig.update_yaxes(gridcolor="#E2E8F0")
st.plotly_chart(fig, width="stretch")

st.markdown("---")
c1, c2 = st.columns([1, 1.1])

with c1:
    st.subheader("Weekday staffing profile")
    dow = series.groupby(series.index.dayofweek).mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig2 = go.Figure(go.Bar(x=names, y=dow.values, marker_color=PALETTE[2]))
    fig2.update_layout(height=300, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
                       yaxis_title="Mean daily revenue (R$)")
    fig2.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig2, width="stretch")
    st.caption(f"Weekends run {(1 - dow.iloc[5:].mean() / dow.iloc[:5].mean()) * 100:.0f}% "
               "below weekdays. This is a rota, directly.")

with c2:
    st.subheader("Model comparison (rolling-origin backtest)")
    bt = backtest[["MAE", "RMSE", "MAPE", "vs_seasonal_naive"]].round(2)
    st.dataframe(bt, width="stretch", height=300)
    st.caption("Only SARIMA beats the seasonal-naive benchmark. Both ML lag "
               "models lose — recursive multi-step forecasting compounds its "
               "own error over 30 steps.")

st.markdown("---")
st.subheader("Daily forecast detail")
detail = fc.copy()
detail.index = detail.index.date
detail.columns = ["Forecast (R$)", "Lower 80% (R$)", "Upper 80% (R$)"]
st.dataframe(detail.round(0), width="stretch", height=280)
st.download_button("Download forecast (CSV)", fc.to_csv(),
                   file_name=f"revenue_forecast_{horizon}d.csv", mime="text/csv")

action_box("Recommended actions", [
    f"<b>Inventory:</b> stock against the upper bound ({brl(fc['upper_80'].sum())}), "
    "not the point forecast. Planning to the mean means stocking out half the time.",
    f"<b>Staffing:</b> the weekday profile is stable — schedule warehouse and "
    f"support capacity to it rather than to a flat daily average.",
    f"<b>Cash:</b> commit spend against the lower bound ({brl(fc['lower_80'].sum())}).",
    "<b>Monitoring:</b> once live, actuals outside the 80% band for two "
    "consecutive days is an alert — payment outage or carrier failure. "
    "Arguably more valuable than the forecast itself.",
])

if horizon >= 90:
    caveat("The 90-day interval crosses zero. That is honest, not broken — "
           "SARIMA uncertainty compounds with horizon and 20 months of history "
           "cannot pin down a quarter. Use 30 days for decisions and treat 90 "
           "as directional only.")

caveat("The series is trimmed 6 days short of 2018-08-31. Olist's data "
       "collection stopped mid-flow, so the final days are right-censored — "
       "revenue falls from ~30k to 1.5k because orders were never recorded. "
       "Left in place, SARIMA read that cliff as trend and forecast negative "
       "revenue.")