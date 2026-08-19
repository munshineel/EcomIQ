"""Inference pipeline: load trained artifacts and score new records.

One entry point per model, plus a single `EcomIQPipeline` that holds them all.
Models are loaded lazily and cached, so importing this module is cheap and only
the models you actually call get read from disk.

The critical contract this enforces: **a model is never used without its
threshold and its feature list.** `predict()` on the churn model defaults to a
0.5 cutoff, which at a 1.8% base rate flags almost nobody. The threshold chosen
on validation is stored inside the artifact and applied here.

Usage
-----
    from src.inference.pipeline import EcomIQPipeline

    pipe = EcomIQPipeline()
    pipe.score_churn(orders_df)          # adds proba + flag columns
    pipe.forecast_revenue(days=30)       # DataFrame with intervals
    pipe.recommend("product_id", k=10)   # list of product ids
    pipe.score_sentiment(["produto ótimo", "não chegou"])
"""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PATHS, PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PATHS["processed"]


class ArtifactMissingError(FileNotFoundError):
    """Raised with the command needed to regenerate the artifact."""


def _load(path: Path, regenerate_with: str):
    import joblib

    if not path.exists():
        raise ArtifactMissingError(
            f"{path.name} not found in {path.parent}.\n"
            f"Regenerate it by running: {regenerate_with}"
        )
    return joblib.load(path)


class EcomIQPipeline:
    """Serving wrapper around the four trained models."""

    def __init__(self, models_dir: Path | None = None,
                 processed_dir: Path | None = None) -> None:
        self.models_dir = models_dir or MODELS_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR

    # ------------------------------------------------------------------ churn
    @cached_property
    def _churn(self) -> dict:
        return _load(self.models_dir / "churn_model.joblib",
                     "python notebooks/04_churn_model.py")

    def score_churn(self, orders: pd.DataFrame, budget_pct: float | None = None
                    ) -> pd.DataFrame:
        """Add `repeat_proba` and `contact_flag` to a first-order frame.

        `budget_pct` overrides the stored threshold with a top-N% cutoff, which
        is how a campaign is actually briefed ("we can contact 5,000 people").
        Without it, the validation-tuned threshold is used.
        """
        bundle = self._churn
        features = bundle["features"]

        missing = [c for c in features if c not in orders.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} required features missing: {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''}\n"
                "Build them with src.features.build.build_churn_features()."
            )

        out = orders.copy()
        out["repeat_proba"] = bundle["model"].predict_proba(out[features])[:, 1]

        if budget_pct is not None:
            threshold = float(np.quantile(out["repeat_proba"], 1 - budget_pct / 100))
        else:
            threshold = float(bundle["threshold"])

        out["contact_flag"] = (out["repeat_proba"] >= threshold).astype(int)
        out.attrs["threshold"] = threshold
        out.attrs["model"] = bundle["model_name"]

        logger.info("scored %d customers | threshold %.4f | flagged %d",
                    len(out), threshold, int(out["contact_flag"].sum()))
        return out

    @property
    def churn_metadata(self) -> dict:
        b = self._churn
        return {"model": b["model_name"], "threshold": b["threshold"],
                "n_features": len(b["features"]), "metrics": b.get("test_metrics", {})}

    # --------------------------------------------------------------- forecast
    def forecast_revenue(self, days: int = 30, use_cached: bool = True) -> pd.DataFrame:
        """Revenue forecast with an 80% interval.

        Reads the precomputed 90-day CSV by default: refitting SARIMA takes
        seconds and the 35MB model file is not worth loading to reproduce a
        result that does not change until new data arrives. Set
        `use_cached=False` to refit.
        """
        if days > 90 and use_cached:
            raise ValueError("Cached forecast covers 90 days. "
                             "Pass use_cached=False to extend it.")

        if use_cached:
            path = self.processed_dir / "revenue_forecast_90d.csv"
            if not path.exists():
                raise ArtifactMissingError(
                    f"{path.name} not found.\n"
                    "Regenerate with: python notebooks/05_forecast.py")
            return pd.read_csv(path, index_col=0, parse_dates=[0]).iloc[:days]

        bundle = _load(self.models_dir / "forecast_model.joblib",
                       "python notebooks/05_forecast.py")
        fc = bundle["model"].get_forecast(steps=days)
        ci = fc.conf_int(alpha=0.20)
        return pd.DataFrame({
            "forecast": fc.predicted_mean,
            "lower_80": ci.iloc[:, 0].to_numpy(),
            "upper_80": ci.iloc[:, 1].to_numpy(),
        })

    # ------------------------------------------------------------ recommender
    @cached_property
    def _recommendations(self) -> pd.DataFrame:
        path = self.processed_dir / "top_recommendations.parquet"
        if not path.exists():
            raise ArtifactMissingError(
                f"{path.name} not found.\n"
                "Regenerate with: python scripts/build_dashboard_data.py")
        return pd.read_parquet(path)

    @cached_property
    def _recommender(self) -> dict:
        return _load(self.models_dir / "recommender.joblib",
                     "python notebooks/06_recommender.py")

    def recommend(self, product_id: str, k: int = 10, live: bool = False) -> list[str]:
        """Top-k similar products.

        Serves from the precomputed table by default (microseconds). Falls back
        to the live model for products outside it -- those with fewer than 5
        sales, which is 85% of the catalogue but a small share of traffic.
        """
        if not live:
            hit = self._recommendations
            hit = hit[hit["seed"] == product_id].nsmallest(k, "rank")
            if len(hit):
                return hit["recommended"].tolist()

        return self._recommender["content"].recommend(product_id, k)

    # -------------------------------------------------------------- sentiment
    @cached_property
    def _sentiment(self) -> dict:
        return _load(self.models_dir / "sentiment_model.joblib",
                     "python notebooks/07_review_nlp.py")

    def score_sentiment(self, texts: list[str] | pd.Series) -> pd.DataFrame:
        """Negative-sentiment score plus aspect flags per comment.

        The score is a probability for LogisticRegression and a signed decision
        value for LinearSVC, so `is_negative` (not the raw score) is the field
        to act on.
        """
        from src.models.sentiment import clean_text, tag_aspects

        texts = pd.Series(texts).astype(str)
        cleaned = texts.map(clean_text)
        model = self._sentiment["model"]

        usable = cleaned.str.split().str.len() >= 2
        out = pd.DataFrame({"text": texts, "clean": cleaned}, index=texts.index)
        out["negative_score"] = np.nan

        if usable.any():
            sub = cleaned[usable]
            out.loc[usable, "negative_score"] = (
                model.predict_proba(sub)[:, 1] if hasattr(model, "predict_proba")
                else model.decision_function(sub)
            )
            out.loc[usable, "is_negative"] = model.predict(sub)

        aspects = tag_aspects(texts)
        return pd.concat([out, aspects.add_prefix("aspect_")], axis=1)

    # ------------------------------------------------------------------ admin
    def health_check(self) -> pd.DataFrame:
        """Report which artifacts exist. Run after a fresh clone."""
        checks = [
            ("churn model", self.models_dir / "churn_model.joblib",
             "python notebooks/04_churn_model.py"),
            ("forecast CSV", self.processed_dir / "revenue_forecast_90d.csv",
             "python notebooks/05_forecast.py"),
            ("recommendations", self.processed_dir / "top_recommendations.parquet",
             "python scripts/build_dashboard_data.py"),
            ("sentiment model", self.models_dir / "sentiment_model.joblib",
             "python notebooks/07_review_nlp.py"),
            ("segments", self.processed_dir / "customer_segments.csv",
             "python notebooks/03_segmentation.py"),
            ("dashboard orders", self.processed_dir / "dashboard_orders.parquet",
             "python scripts/build_dashboard_data.py"),
        ]
        return pd.DataFrame([
            {"artifact": name, "present": path.exists(),
             "size_mb": round(path.stat().st_size / 1e6, 2) if path.exists() else 0.0,
             "regenerate_with": cmd if not path.exists() else ""}
            for name, path, cmd in checks
        ])


__all__ = ["EcomIQPipeline", "ArtifactMissingError"]