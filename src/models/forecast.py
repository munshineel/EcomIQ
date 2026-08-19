"""Daily revenue forecasting: series construction, features, backtesting.

Design rule: every function here is causal. Nothing uses information dated
later than the point being predicted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ANALYSIS_END, ANALYSIS_START

logger = logging.getLogger(__name__)

SEASONAL_PERIOD = 7          # weekly, validated in EDA E13
DEFAULT_HORIZON = 30
LAGS = (1, 2, 3, 7, 14, 21, 28)
ROLL_WINDOWS = (7, 14, 28)


def trim_incomplete_tail(s: pd.Series, frac: float = 0.6) -> pd.Series:
    """Drop trailing days whose revenue is implausibly low.

    Olist's data collection stopped mid-flow, so the final ~10 days are
    right-censored: revenue falls from ~30k to 1.5k because orders were not yet
    recorded, not because sales dropped. Left in place, SARIMA reads that cliff
    as trend and forecasts NEGATIVE revenue.

    Rule: walk back from the end while the individual day sits below
    `frac` x median. A trailing 7-day mean does NOT work here -- it keeps
    averaging in the healthy days just before the cliff and never trips.
    frac=0.6 is safely below legitimate lows: weekends run at ~0.83 x median.
    """
    med = s.median()
    cut = len(s)
    while cut > 7 and s.iloc[cut - 1] < frac * med:
        cut -= 1
    if cut < len(s):
        logger.info("trimmed %d right-censored days (last kept: %s)",
                    len(s) - cut, s.index[cut - 1].date())
    return s.iloc[:cut]


def build_daily_revenue(order_frame: pd.DataFrame, trim_tail: bool = True) -> pd.Series:
    """Daily net product revenue from fulfilled orders, gap-filled."""
    df = order_frame[order_frame["is_fulfilled"] & order_frame["revenue"].notna()]
    s = (
        df.set_index("order_purchase_timestamp")["revenue"]
        .resample("D").sum().asfreq("D").fillna(0.0)
    )
    s = s.loc[ANALYSIS_START:ANALYSIS_END]
    if trim_tail:
        s = trim_incomplete_tail(s)
    s.name = "revenue"
    logger.info("daily series: %d days, %d zero days, median %.0f",
                len(s), int((s == 0).sum()), s.median())
    return s


def make_supervised(series: pd.Series) -> pd.DataFrame:
    """Turn the series into a lag/rolling feature table.

    Every rolling statistic is `.shift(1)`-ed first. Without that shift the
    window includes the day being predicted, so the feature contains the
    target -- the most common time-series leak there is.
    """
    df = pd.DataFrame({"y": series})

    for lag in LAGS:
        df[f"lag_{lag}"] = series.shift(lag)

    shifted = series.shift(1)                      # <- the causality guard
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"roll_std_{w}"] = shifted.rolling(w).std()
    df["roll_min_7"] = shifted.rolling(7).min()
    df["roll_max_7"] = shifted.rolling(7).max()

    idx = series.index
    df["dow"] = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["day_of_month"] = idx.day
    df["month"] = idx.month
    df["week_of_year"] = idx.isocalendar().week.to_numpy()
    df["time_index"] = np.arange(len(df))          # linear trend

    return df.dropna()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error, computed only where y > 0.

    Four days in the series have zero revenue, and MAPE divides by y -- those
    days would return infinity. Excluding them is standard, but it means MAPE
    is not measuring the full series, which is why MAE is the headline here.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y > 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100)


def smape(y, yhat) -> float:
    """Symmetric MAPE: bounded, and safe when y is zero."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    m = denom > 0
    return float(np.mean(np.abs(y[m] - yhat[m]) / denom[m]) * 100)


def score(y, yhat) -> dict[str, float]:
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat),
            "MAPE": mape(y, yhat), "sMAPE": smape(y, yhat)}


# ---------------------------------------------------------------------------
# rolling-origin backtest
# ---------------------------------------------------------------------------
def rolling_origin_folds(
    n: int, horizon: int = DEFAULT_HORIZON, n_folds: int = 5, min_train: int = 180
) -> list[tuple[int, int]]:
    """Expanding-window fold boundaries as (train_end, test_end) index pairs.

    Train always ends immediately before test begins, and test is exactly
    `horizon` long. Expanding rather than sliding because more history helps
    and we only have 606 days.
    """
    folds = []
    last_start = n - horizon
    step = max(1, (last_start - min_train) // max(1, n_folds - 1))
    for i in range(n_folds):
        train_end = min_train + i * step
        if train_end + horizon > n:
            break
        folds.append((train_end, train_end + horizon))
    return folds


__all__ = [
    "SEASONAL_PERIOD", "DEFAULT_HORIZON", "LAGS", "ROLL_WINDOWS",
    "build_daily_revenue", "trim_incomplete_tail", "make_supervised",
    "mae", "rmse", "mape", "smape", "score", "rolling_origin_folds",
]