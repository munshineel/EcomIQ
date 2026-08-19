"""Recommenders and ranking metrics for the Olist catalogue.

Design forced by the data: 55% of products sold exactly once and only 3.28% of
orders contain two distinct products. Content-based similarity is therefore the
primary system; collaborative filtering is implemented as a benchmark and is
expected to lose. Both are evaluated on the same held-out co-purchase pairs.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

logger = logging.getLogger(__name__)

NUMERIC_ATTRS = ["price", "product_weight_g", "volume_cm3", "product_photos_qty"]


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------
def build_product_catalog(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per product: category, median price, physical attributes, sales."""
    products = tables["products"].copy()
    items = tables["order_items"]

    stats = items.groupby("product_id").agg(
        price=("price", "median"),
        units_sold=("order_item_id", "count"),
        n_orders=("order_id", "nunique"),
    )

    cat = products.set_index("product_id").join(stats, how="inner")
    cat["volume_cm3"] = (
        cat["product_length_cm"] * cat["product_height_cm"] * cat["product_width_cm"]
    )
    cat["product_category_name"] = cat["product_category_name"].fillna("unknown")
    for col in NUMERIC_ATTRS:
        cat[col] = cat[col].fillna(cat[col].median())

    logger.info("catalog: %d products | %d sold once (%.1f%%)",
                len(cat), int((cat.units_sold == 1).sum()),
                (cat.units_sold == 1).mean() * 100)
    return cat


def build_copurchase_pairs(tables: dict[str, pd.DataFrame],
                           order_frame: pd.DataFrame) -> pd.DataFrame:
    """Every ordered (a, b) pair of distinct products bought in the same order.

    This is the only ground truth available. Directed pairs (both a->b and
    b->a) because at query time we know one item and must rank the other.
    """
    items = tables["order_items"][["order_id", "product_id"]].drop_duplicates()
    dates = order_frame.set_index("order_id")["order_purchase_timestamp"]

    rows = []
    for order_id, group in items.groupby("order_id")["product_id"]:
        prods = list(dict.fromkeys(group))
        if len(prods) < 2:
            continue
        ts = dates.get(order_id)
        for i, a in enumerate(prods):
            for b in prods[i + 1:]:
                rows.append((order_id, a, b, ts))
                rows.append((order_id, b, a, ts))

    pairs = pd.DataFrame(rows, columns=["order_id", "seed", "target", "ts"])
    logger.info("co-purchase pairs: %d directed pairs from %d orders",
                len(pairs), pairs.order_id.nunique())
    return pairs.dropna(subset=["ts"])


# ---------------------------------------------------------------------------
# recommenders
# ---------------------------------------------------------------------------
class PopularityRecommender:
    """Baseline: always return the global best-sellers.

    Deliberately included. It has zero personalisation and terrible coverage,
    yet on sparse data it is often hard to beat -- so any model that cannot
    beat it is not worth deploying.
    """

    name = "popularity"

    def fit(self, catalog: pd.DataFrame, pairs: pd.DataFrame | None = None):
        self.top_ = catalog.sort_values("units_sold", ascending=False).index.to_numpy()
        return self

    def recommend(self, seed: str, k: int = 10) -> list[str]:
        return [p for p in self.top_[: k + 1] if p != seed][:k]


class CategoryPopularityRecommender:
    """Best-sellers within the seed item's own category.

    A smarter baseline: uses one attribute of the seed, nothing more.
    """

    name = "category_popularity"

    def fit(self, catalog: pd.DataFrame, pairs: pd.DataFrame | None = None):
        self.catalog_ = catalog
        self.by_cat_ = {
            c: g.sort_values("units_sold", ascending=False).index.to_numpy()
            for c, g in catalog.groupby("product_category_name", observed=True)
        }
        self.global_ = catalog.sort_values("units_sold", ascending=False).index.to_numpy()
        return self

    def recommend(self, seed: str, k: int = 10) -> list[str]:
        if seed not in self.catalog_.index:
            return list(self.global_[:k])
        cat = self.catalog_.at[seed, "product_category_name"]
        cands = [p for p in self.by_cat_.get(cat, [])[: k + 1] if p != seed][:k]
        if len(cands) < k:
            cands += [p for p in self.global_ if p != seed and p not in cands][:k - len(cands)]
        return cands


class ContentRecommender:
    """Cosine similarity over category (one-hot) + rank-normalised attributes.

    QuantileTransformer rather than StandardScaler: price, weight and volume are
    heavily right-skewed, so rank-normalising puts them on a comparable footing
    without a handful of 30kg items dominating the distance.

    `category_weight` controls how much of the similarity is "same category".
    At 1.0 the model is close to a category filter; lower values let it cross
    category boundaries on physical similarity.
    """

    name = "content"

    def __init__(self, category_weight: float = 3.0, n_neighbors: int = 50):
        self.category_weight = category_weight
        self.n_neighbors = n_neighbors

    def fit(self, catalog: pd.DataFrame, pairs: pd.DataFrame | None = None):
        self.catalog_ = catalog
        self.index_ = catalog.index.to_numpy()
        self.pos_ = {p: i for i, p in enumerate(self.index_)}

        cats = OneHotEncoder(handle_unknown="ignore").fit_transform(
            catalog[["product_category_name"]]
        )
        nums = QuantileTransformer(
            output_distribution="normal", n_quantiles=min(1000, len(catalog)),
            random_state=42,
        ).fit_transform(catalog[NUMERIC_ATTRS])

        X = sparse.hstack(
            [cats * self.category_weight, sparse.csr_matrix(nums)]
        ).tocsr()
        self.nn_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(catalog)),
            metric="cosine", algorithm="brute",
        ).fit(X)
        self.X_ = X
        return self

    def recommend(self, seed: str, k: int = 10) -> list[str]:
        i = self.pos_.get(seed)
        if i is None:
            return []
        _, idx = self.nn_.kneighbors(self.X_[i], n_neighbors=min(k + 1, len(self.index_)))
        return [self.index_[j] for j in idx[0] if self.index_[j] != seed][:k]

    def scores(self, seed: str, k: int = 50) -> dict[str, float]:
        i = self.pos_.get(seed)
        if i is None:
            return {}
        dist, idx = self.nn_.kneighbors(self.X_[i], n_neighbors=min(k + 1, len(self.index_)))
        return {self.index_[j]: 1.0 - d
                for d, j in zip(dist[0], idx[0]) if self.index_[j] != seed}


class CoOccurrenceRecommender:
    """Item-item collaborative filtering from basket co-occurrence.

    Fitted ONLY on training pairs. Falls back to category popularity when the
    seed has no co-occurrence history -- which, given 55% of products sold once,
    is the overwhelming majority of queries. That fallback rate is the headline
    finding, not a workaround.
    """

    name = "collaborative"

    def fit(self, catalog: pd.DataFrame, pairs: pd.DataFrame):
        self.fallback_ = CategoryPopularityRecommender().fit(catalog)
        counts: dict[str, Counter] = defaultdict(Counter)
        for seed, target in zip(pairs["seed"], pairs["target"]):
            counts[seed][target] += 1
        self.counts_ = counts
        self.n_seeds_ = len(counts)
        logger.info("co-occurrence: %d seed products have any history (%.1f%% of catalog)",
                    self.n_seeds_, self.n_seeds_ / len(catalog) * 100)
        return self

    def recommend(self, seed: str, k: int = 10) -> list[str]:
        hist = self.counts_.get(seed)
        if not hist:
            return self.fallback_.recommend(seed, k)
        recs = [p for p, _ in hist.most_common(k)]
        if len(recs) < k:
            recs += [p for p in self.fallback_.recommend(seed, k) if p not in recs][:k - len(recs)]
        return recs[:k]

    def scores(self, seed: str, k: int = 50) -> dict[str, float]:
        hist = self.counts_.get(seed)
        if not hist:
            return {}
        total = sum(hist.values())
        return {p: c / total for p, c in hist.most_common(k)}


class HybridRecommender:
    """Weighted score fusion of content, co-occurrence and popularity.

    For a seed item s and candidate c:

        score(c) = w_cf   * cooccur_norm(s, c)
                 + w_cont * content_sim(s, c)
                 + w_pop  * popularity_norm(c)

    Each component is normalised to [0, 1] before weighting, otherwise the
    weights would be meaningless -- cosine similarity lives in [0, 1] while raw
    co-occurrence counts are unbounded integers.

    Co-occurrence gets the largest weight because when it fires it is direct
    behavioural evidence. It simply almost never fires here, so in practice most
    scores come from content plus popularity. Popularity is kept at a small
    weight as a tie-breaker among equally similar items.
    """

    name = "hybrid"

    def __init__(self, w_cf: float = 0.5, w_content: float = 0.4, w_pop: float = 0.1,
                 category_weight: float = 3.0):
        self.w_cf, self.w_content, self.w_pop = w_cf, w_content, w_pop
        self.category_weight = category_weight

    def fit(self, catalog: pd.DataFrame, pairs: pd.DataFrame):
        self.content_ = ContentRecommender(self.category_weight).fit(catalog)
        self.cf_ = CoOccurrenceRecommender().fit(catalog, pairs)
        pop = catalog["units_sold"].astype(float)
        self.pop_ = (pop.rank(pct=True)).to_dict()
        self.fallback_ = CategoryPopularityRecommender().fit(catalog)
        return self

    def recommend(self, seed: str, k: int = 10) -> list[str]:
        cf = self.cf_.scores(seed, k=50)
        content = self.content_.scores(seed, k=50)
        if not cf and not content:
            return self.fallback_.recommend(seed, k)

        cf_max = max(cf.values()) if cf else 1.0
        combined: dict[str, float] = {}
        for cand in set(cf) | set(content):
            if cand == seed:
                continue
            combined[cand] = (
                self.w_cf * (cf.get(cand, 0.0) / cf_max)
                + self.w_content * content.get(cand, 0.0)
                + self.w_pop * self.pop_.get(cand, 0.0)
            )
        ranked = sorted(combined, key=combined.get, reverse=True)[:k]
        if len(ranked) < k:
            ranked += [p for p in self.fallback_.recommend(seed, k) if p not in ranked][:k - len(ranked)]
        return ranked[:k]


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------
def precision_at_k(recs: list[str], relevant: set[str], k: int) -> float:
    """Of the k items shown, what fraction were relevant?

    With exactly one relevant item per query the ceiling is 1/k, so P@10 can
    never exceed 0.10. Read it relative to that ceiling, not against 1.0.
    """
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list[str], relevant: set[str], k: int) -> float:
    """Of the relevant items, what fraction appeared in the top k?

    With one relevant item this equals hit-rate: did we find it, yes or no.
    """
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def average_precision_at_k(recs: list[str], relevant: set[str], k: int) -> float:
    """Precision averaged at each position where a relevant item appears.

    Rewards putting relevant items EARLY, which precision@k ignores.
    """
    hits, score = 0, 0.0
    for i, r in enumerate(recs[:k], start=1):
        if r in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(recs: list[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain, normalised.

    Gain is discounted by log2(position + 1): a hit at rank 1 is worth ~1.0, at
    rank 10 about 0.29. The smoothest position-aware metric of the four.
    """
    dcg = sum(1.0 / np.log2(i + 1) for i, r in enumerate(recs[:k], start=1) if r in relevant)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_recommender(
    model, test_pairs: pd.DataFrame, catalog: pd.DataFrame, k: int = 10,
    max_queries: int | None = None,
) -> dict[str, float]:
    """Score a recommender on held-out co-purchase pairs.

    Also reports COVERAGE: the share of the catalogue that ever appears in any
    recommendation. A model with strong hit-rate and 0.1% coverage is a
    best-seller list wearing a costume.
    """
    truth = test_pairs.groupby("seed")["target"].apply(set)
    if max_queries and len(truth) > max_queries:
        truth = truth.sample(max_queries, random_state=42)

    p, r, ap, nd = [], [], [], []
    shown: set[str] = set()
    n_empty = 0

    for seed, relevant in truth.items():
        recs = model.recommend(seed, k)
        if not recs:
            n_empty += 1
            continue
        shown.update(recs)
        p.append(precision_at_k(recs, relevant, k))
        r.append(recall_at_k(recs, relevant, k))
        ap.append(average_precision_at_k(recs, relevant, k))
        nd.append(ndcg_at_k(recs, relevant, k))

    return {
        "model": model.name,
        f"precision@{k}": float(np.mean(p)) if p else 0.0,
        f"recall@{k}": float(np.mean(r)) if r else 0.0,
        f"MAP@{k}": float(np.mean(ap)) if ap else 0.0,
        f"NDCG@{k}": float(np.mean(nd)) if nd else 0.0,
        "coverage": len(shown) / len(catalog),
        "queries": len(p),
        "no_recs": n_empty,
    }


__all__ = [
    "build_product_catalog", "build_copurchase_pairs",
    "PopularityRecommender", "CategoryPopularityRecommender",
    "ContentRecommender", "CoOccurrenceRecommender", "HybridRecommender",
    "precision_at_k", "recall_at_k", "average_precision_at_k", "ndcg_at_k",
    "evaluate_recommender", "NUMERIC_ATTRS",
]