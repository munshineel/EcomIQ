# Architecture

```
                    ┌──────────────────────────────┐
                    │      9 raw Olist CSVs        │
                    │    99k orders, 2017–2018     │
                    └──────────────┬───────────────┘
                                   ↓
              ┌────────────────────────────────────────┐
              │              src/data                  │
              │      schema · load · prepare           │
              │  ┌──────────────────────────────────┐  │
              │  │ order_frame   (1 row / order)    │  │
              │  │ customer_frame (1 row / person)  │  │
              │  └──────────────────────────────────┘  │
              └────────────────────┬───────────────────┘
                                   ↓
              ┌────────────────────────────────────────┐
              │            src/features                │
              │   two targets · time splits ·          │
              │   leakage guard                        │
              └────────────────────┬───────────────────┘
                                   ↓
      ╔════════════════════════════════════════════════════════╗
      ║                    src/models                          ║
      ║  ┌────────────┐  ┌────────────┐  ┌────────────┐        ║
      ║  │Segmentation│  │   Churn    │  │  Forecast  │        ║
      ║  │k-means, PCA│  │XGBoost,SHAP│  │  SARIMA    │        ║
      ║  └────────────┘  └────────────┘  └────────────┘        ║
      ║        ┌──────────────┐  ┌──────────────┐              ║
      ║        │ Recommender  │  │  Sentiment   │              ║
      ║        │content-based │  │TF-IDF,aspects│              ║
      ║        └──────────────┘  └──────────────┘              ║
      ╚════════════════════════════╤═══════════════════════════╝
                                   ↓
              ┌────────────────────────────────────────┐
              │           src/inference                 │
              │   artifacts + thresholds + features    │
              └────────────────────┬───────────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │    Streamlit · 8 pages       │
                    └──────────────────────────────┘
```

## Design rules

**Grain is explicit in every function name.** `build_order_frame()` returns one
row per `order_id`; `build_customer_frame()` returns one row per
`customer_unique_id`. Joining the two without aggregating would silently
double-count revenue, and nothing would raise. Tests assert both grains.

**One direction only.** Each layer writes an artifact the next layer reads, so
any stage can be re-run without re-running the whole chain. Notebooks import
from `src/`; `src/` never imports from a notebook.

**Thresholds travel with models.** `models/churn_model.joblib` stores the
model, its feature list, and the threshold tuned on validation. Calling
`predict()` without it defaults to a 0.5 cutoff, which at a 1.8% base rate
flags almost nobody.

**No random splits anywhere.** Every target is either a future event or sits in
a time-ordered stream.

| Model | Split | Reason |
|---|---|---|
| Churn | chronological 60/20/20 | target is a future event; threshold tuned on val so test stays clean |
| Forecast | 5-fold rolling origin | one split is a single lucky estimate |
| Recommender | co-purchase pairs by time | pairs are stored both directions — a random split leaks the answer |
| Sentiment | chronological by review date | vocabulary and complaint mix drift |