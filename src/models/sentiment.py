"""Portuguese review sentiment and aspect analysis.

Labels come free from `review_score`, so sentiment here is SUPERVISED and
properly evaluable -- no lexicon guessing.

Corpus facts that shape every choice below: 40,977 comments, median 9 words,
maximum 208 characters, Portuguese, and 55% contain accented characters.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Portuguese stopwords, hardcoded rather than pulled from NLTK. Reasons: no
# runtime nltk.download() that can fail on a fresh clone, works offline, and
# fully reproducible. sklearn ships English stopwords only.
PT_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as",
    "até", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "foram", "há", "isso", "isto", "já", "lhe", "lhes",
    "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na",
    "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num",
    "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por",
    "qual", "quando", "que", "quem", "se", "sem", "ser", "seu", "seus", "só",
    "sua", "suas", "também", "te", "tem", "tém", "tenho", "ter", "teu", "teus",
    "tu", "tua", "tuas", "um", "uma", "você", "vocês", "vos", "à", "às", "é",
    "está", "estão", "esteja", "eles", "aí", "ainda", "agora", "então", "pois",
    "sobre", "todo", "toda", "todos", "todas", "isso", "onde",
    # Olist anonymised real partner names by substituting Game of Thrones
    # houses. These are artefacts of the release, not customer vocabulary, and
    # would otherwise show up as top features.
    "lannister", "stark", "targaryen", "baratheon", "tyrell", "greyjoy",
    "arryn", "martell", "tully", "bolton", "frey", "mormont",
}

# Negation words kept OUT of the stopword list on purpose: "não recomendo"
# (do not recommend) inverts sentiment, and dropping "não" destroys it.
NEGATIONS = {"não", "nao", "nunca", "jamais", "nada", "nenhum", "nenhuma"}
PT_STOPWORDS -= NEGATIONS

# Aspects DERIVED FROM THE CORPUS, not invented. Each pattern below was chosen
# after ranking the 60 most frequent tokens in the actual comments and checking
# coverage and score separation. Coverage / mean score measured on 40,977
# comments is recorded beside each.
ASPECT_PATTERNS: dict[str, str] = {
    # 55.0% of comments, mean 3.48 stars
    "delivery": r"entreg|prazo|chegou|receb|correio|previst|atras|demor|rápid|frete|envio|transport",
    # 13.5%, mean 3.37
    "seller_service": r"loja|vendedor|atendiment|contato|respond|suporte|site",
    # 12.4%, mean 3.79
    "product_quality": r"qualidade|resistent|frágil|quebr|defeit|danific|material|péssim",
    # 8.2%, mean 1.71  <- the most damaging aspect in the corpus
    "completeness": r"faltou|falta|incomplet|só vei|apenas um|não vei|não receb",
    # 7.8%, mean 3.27
    "expectation_match": r"conforme|igual|descri|diferente|errad|não era|esperava|foto",
    # 4.6%, mean 3.68
    "packaging": r"embal|caixa|lacrad|protegid|amass",
    # 4.4%, mean 3.59
    "price_value": r"preç|barat|car[oa]\b|custo|valor|vale a pena|promoç",
}


# ---------------------------------------------------------------------------
# text cleaning
# ---------------------------------------------------------------------------
def clean_text(s: str, strip_accents: bool = False) -> str:
    """Lowercase, strip URLs/digits/punctuation, collapse whitespace.

    Deliberately minimal. Measured on this corpus: 0 comments contain HTML
    tags, 6 contain English contractions, 273 (0.67%) contain emoji. Importing
    beautifulsoup, `contractions` or the `emoji` package to handle that would
    add dependencies for nothing -- the regexes below cover it.

    `strip_accents=False` by default because 55% of comments carry accents and
    they are meaning-bearing in Portuguese.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]", " ", s)
    s = re.sub(r"\d+", " ", s)
    if strip_accents:
        s = "".join(c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-záàâãéêíóôõúüç\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str, remove_stopwords: bool = True, min_len: int = 3) -> list[str]:
    """Whitespace tokenisation after cleaning.

    Sufficient here: Portuguese is space-delimited and comments average 9
    words. A spaCy pipeline would add a dependency parser and a 50MB model
    download to split strings on spaces.
    """
    tokens = clean_text(s).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in PT_STOPWORDS]
    return [t for t in tokens if len(t) >= min_len or t in NEGATIONS]


# ---------------------------------------------------------------------------
# supervised dataset
# ---------------------------------------------------------------------------
def build_sentiment_dataset(
    order_reviews: pd.DataFrame, drop_neutral: bool = True, min_chars: int = 10
) -> pd.DataFrame:
    """Comments with a binary sentiment label from review_score.

    1-2 stars -> negative (1), 4-5 stars -> positive (0), 3 stars dropped.
    Dropping neutrals is standard: a 3-star review is genuinely ambiguous, and
    including it as a third class adds a category the business cannot act on.
    """
    df = order_reviews.dropna(subset=["review_comment_message"]).copy()
    df["text"] = df["review_comment_message"].astype(str)
    df = df[df["text"].str.len() >= min_chars]

    if drop_neutral:
        df = df[df["review_score"] != 3]
    df["label"] = (df["review_score"] <= 2).astype(int)   # 1 = negative
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.split().str.len() >= 2]

    logger.info("sentiment dataset: %d rows | negative %.1f%%",
                len(df), df["label"].mean() * 100)
    return df.sort_values("review_creation_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# aspect extraction
# ---------------------------------------------------------------------------
def tag_aspects(texts: pd.Series) -> pd.DataFrame:
    """Boolean column per aspect: does the comment mention it?

    Keyword matching rather than a model. Justification: median comment is 9
    words. There is not enough context for a dependency parser or an
    aspect-extraction transformer to beat regex, and keywords stay auditable --
    a business user can read the pattern and disagree with it.
    """
    cleaned = texts.map(clean_text)
    return pd.DataFrame(
        {a: cleaned.str.contains(p, regex=True) for a, p in ASPECT_PATTERNS.items()},
        index=texts.index,
    )


def aspect_sentiment_summary(df: pd.DataFrame, score_col: str = "review_score") -> pd.DataFrame:
    """Per aspect: mention volume, mean score, negative share.

    This is aspect-based sentiment: not "this review is negative" but "this
    review is negative ABOUT DELIVERY", which is what routes to a team.
    """
    flags = tag_aspects(df["review_comment_message"].fillna(""))
    rows = []
    for aspect in ASPECT_PATTERNS:
        m = flags[aspect]
        sub = df[m]
        if len(sub) == 0:
            continue
        rows.append({
            "aspect": aspect,
            "mentions": int(m.sum()),
            "share_of_comments": float(m.mean()),
            "mean_score": float(sub[score_col].mean()),
            "pct_negative": float((sub[score_col] <= 2).mean()),
        })
    out = pd.DataFrame(rows).set_index("aspect").sort_values("pct_negative", ascending=False)
    out["score_vs_overall"] = (out["mean_score"] - df[score_col].mean()).round(3)
    return out


__all__ = [
    "PT_STOPWORDS", "NEGATIONS", "ASPECT_PATTERNS",
    "clean_text", "tokenize", "build_sentiment_dataset",
    "tag_aspects", "aspect_sentiment_summary",
]