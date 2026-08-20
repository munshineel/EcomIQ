"""Inference pipeline: artifact contracts and serving behaviour."""

from __future__ import annotations

import pandas as pd
import pytest

from src.inference.pipeline import ArtifactMissingError, PropheticIQPipeline
from tests.conftest import needs_raw

pipe = PropheticIQPipeline()
ARTIFACTS_PRESENT = pipe.health_check()["present"].all()
needs_artifacts = pytest.mark.skipif(
    not ARTIFACTS_PRESENT, reason="model artifacts not built; run the notebooks first")


def test_health_check_reports_every_artifact():
    hc = pipe.health_check()
    assert len(hc) == 6
    assert set(hc.columns) == {"artifact", "present", "size_mb", "regenerate_with"}


def test_missing_artifact_error_names_the_fix(tmp_path):
    broken = PropheticIQPipeline(models_dir=tmp_path, processed_dir=tmp_path)
    with pytest.raises(ArtifactMissingError, match="notebooks/04_churn_model.py"):
        _ = broken.churn_metadata


@needs_artifacts
def test_churn_threshold_is_stored_with_the_model():
    """A model without its threshold is unusable: predict() defaults to 0.5,
    which at a 1.8% base rate flags almost nobody."""
    meta = pipe.churn_metadata
    assert 0.0 < meta["threshold"] < 1.0
    assert meta["n_features"] == 28


@needs_artifacts
def test_score_churn_rejects_a_frame_missing_features():
    with pytest.raises(ValueError, match="features missing"):
        pipe.score_churn(pd.DataFrame({"revenue": [10.0]}))


@needs_artifacts
def test_score_churn_budget_flags_the_requested_share():
    churn = pd.read_parquet("data/processed/churn_features.parquet").tail(2000)
    scored = pipe.score_churn(churn, budget_pct=5)
    assert scored["contact_flag"].sum() == pytest.approx(100, abs=5)
    assert scored["repeat_proba"].between(0, 1).all()


@needs_artifacts
def test_forecast_returns_ordered_intervals():
    fc = pipe.forecast_revenue(30)
    assert len(fc) == 30
    assert (fc["lower_80"] <= fc["forecast"]).all()
    assert (fc["forecast"] <= fc["upper_80"]).all()


@needs_artifacts
def test_forecast_beyond_cache_raises():
    with pytest.raises(ValueError, match="90 days"):
        pipe.forecast_revenue(200)


@needs_artifacts
def test_recommend_returns_k_distinct_products_excluding_the_seed():
    seed = pd.read_parquet("data/processed/top_recommendations.parquet")["seed"].iloc[0]
    recs = pipe.recommend(seed, k=5)
    assert len(recs) == 5
    assert len(set(recs)) == 5
    assert seed not in recs


@needs_artifacts
def test_sentiment_scores_and_tags_aspects():
    out = pipe.score_sentiment([
        "produto ótimo chegou antes do prazo",
        "não recebi o produto faltou item",
    ])
    assert len(out) == 2
    assert out["aspect_delivery"].all()
    assert out.loc[1, "aspect_completeness"]
    # The complaint must score more negative than the compliment.
    assert out.loc[1, "negative_score"] > out.loc[0, "negative_score"]


@needs_artifacts
def test_sentiment_handles_unusable_text():
    out = pipe.score_sentiment(["", "ok"])
    assert out["negative_score"].isna().any()