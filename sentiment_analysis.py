"""
Twitter Sentiment Analysis — End-to-End Pipeline
==================================================
Cleans raw tweets, runs exploratory analysis, trains and compares three
classifiers, then builds a self-contained interactive HTML dashboard.

Run:
    python sentiment_analysis.py

Inputs (same folder, or edit the paths below):
    twitter_training.csv
    twitter_validation.csv

Outputs:
    twitter_clean.csv
    stats.json
    ml_results.json
    twitter_sentiment_dashboard.html
"""

import json
import re
import string
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TRAIN_PATH = Path("twitter_training.csv")
VAL_PATH = Path("twitter_validation.csv")
OUT_CLEAN_CSV = Path("twitter_clean.csv")
OUT_STATS_JSON = Path("stats.json")
OUT_ML_JSON = Path("ml_results.json")
OUT_DASHBOARD_HTML = Path("twitter_sentiment_dashboard.html")

COLUMN_NAMES = ["tweet_id", "entity", "sentiment", "text"]
VALID_SENTIMENTS = ["Positive", "Negative", "Neutral", "Irrelevant"]
TOP_N_WORDS = 25
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Optional NLTK support (lemmatisation). Falls back gracefully if the
# nltk corpora aren't available / there's no internet access, so the
# pipeline still runs end to end using sklearn's built-in stopword list
# and un-lemmatised tokens.
# --------------------------------------------------------------------------
_LEMMATIZER = None
_STOPWORDS = set(ENGLISH_STOP_WORDS)

def _try_load_nltk():
    global _LEMMATIZER, _STOPWORDS
    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords
        from nltk.stem import WordNetLemmatizer

        for pkg in ["stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.data.find(
                    f"corpora/{pkg}" if pkg != "omw-1.4" else "corpora/omw-1.4"
                )
            except LookupError:
                nltk.download(pkg, quiet=True)

        _STOPWORDS = set(nltk_stopwords.words("english"))
        _LEMMATIZER = WordNetLemmatizer()
        print("[nltk] stopwords + WordNet lemmatiser loaded.")
    except Exception as e:  # noqa: BLE001
        print(f"[nltk] unavailable ({e}); falling back to sklearn stopwords "
              f"and un-lemmatised tokens.")


_try_load_nltk()

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/punctuation/numbers, drop stopwords,
    lemmatise (if available), return a cleaned string of tokens."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub(" ", text)  # keep the word, drop '#'
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = [t for t in text.split() if t not in _STOPWORDS and len(t) > 2]

    if _LEMMATIZER is not None:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# --------------------------------------------------------------------------
# Load + clean
# --------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, encoding="utf-8")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["text", "sentiment"])
    df = df[df["sentiment"].isin(VALID_SENTIMENTS)]
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].apply(clean_text)
    df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))
    df = df[df["word_count"] > 0]
    df = df.drop_duplicates(subset=["clean_text", "sentiment", "entity"])
    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
def top_words_by_sentiment(df: pd.DataFrame, n: int = TOP_N_WORDS) -> dict:
    result = {}
    for sentiment in VALID_SENTIMENTS:
        words = " ".join(df.loc[df["sentiment"] == sentiment, "clean_text"]).split()
        counts = Counter(words).most_common(n)
        result[sentiment] = [{"text": w, "count": c} for w, c in counts]
    return result


def run_eda(df: pd.DataFrame) -> dict:
    sentiment_counts = df["sentiment"].value_counts().reindex(VALID_SENTIMENTS).fillna(0).astype(int)

    entity_counts = df["entity"].value_counts()
    top_entities = entity_counts.head(15)

    # sentiment breakdown per entity (for heatmap), limited to top entities for readability
    pivot = (
        df[df["entity"].isin(top_entities.index)]
        .groupby(["entity", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=VALID_SENTIMENTS, fill_value=0)
    )
    # normalise each row to % share of that entity's tweets
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).round(4) * 100
    pivot_pct = pivot_pct.reindex(top_entities.index)  # keep volume order

    word_length_stats = (
        df.groupby("sentiment")["word_count"]
        .mean()
        .reindex(VALID_SENTIMENTS)
        .round(2)
        .fillna(0)
    )

    stats = {
        "total_tweets_raw": None,  # filled in by caller
        "total_tweets_clean": int(len(df)),
        "num_entities": int(df["entity"].nunique()),
        "sentiment_counts": sentiment_counts.to_dict(),
        "sentiment_pct": (sentiment_counts / sentiment_counts.sum() * 100).round(2).to_dict(),
        "top_entities": top_entities.to_dict(),
        "entity_sentiment_heatmap": {
            "entities": list(pivot_pct.index),
            "sentiments": VALID_SENTIMENTS,
            "matrix": pivot_pct.values.tolist(),
        },
        "avg_word_count_by_sentiment": word_length_stats.to_dict(),
        "top_words_by_sentiment": top_words_by_sentiment(df),
    }
    return stats


# --------------------------------------------------------------------------
# ML
# --------------------------------------------------------------------------
def run_ml(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    train_df = train_df[train_df["clean_text"].str.len() > 0]
    val_df = val_df[val_df["clean_text"].str.len() > 0]

    X_train_text, y_train = train_df["clean_text"], train_df["sentiment"]
    X_val_text, y_val = val_df["clean_text"], val_df["sentiment"]

    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }

    labels = VALID_SENTIMENTS
    results = {"models": {}, "labels": labels, "vocab_size": len(vectorizer.vocabulary_)}

    best_name, best_acc = None, -1
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average="weighted", zero_division=0)
        rec = recall_score(y_val, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_val, preds, labels=labels)

        results["models"][name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
        }

        if acc > best_acc:
            best_acc, best_name = acc, name

    results["best_model"] = best_name
    return results


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def build_dashboard(stats: dict, ml: dict, out_path: Path) -> None:
    template_path = Path(__file__).with_name("dashboard_template.html")
    template = template_path.read_text(encoding="utf-8")
    payload = json.dumps({"stats": stats, "ml": ml})
    html = template.replace("__DASHBOARD_DATA__", payload)
    out_path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading raw data...")
    train_raw = load_raw(TRAIN_PATH)
    val_raw = load_raw(VAL_PATH)
    total_raw = len(train_raw) + len(val_raw)

    print("Cleaning + lemmatising tweets...")
    train_clean = clean_dataframe(train_raw)
    val_clean = clean_dataframe(val_raw)

    full_clean = pd.concat(
        [train_clean.assign(split="train"), val_clean.assign(split="validation")],
        ignore_index=True,
    )
    full_clean.to_csv(OUT_CLEAN_CSV, index=False)
    print(f"Saved cleaned data -> {OUT_CLEAN_CSV} ({len(full_clean)} rows)")

    print("Running EDA...")
    stats = run_eda(full_clean)
    stats["total_tweets_raw"] = int(total_raw)
    with open(OUT_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved EDA stats -> {OUT_STATS_JSON}")

    print("Training + evaluating models (this may take a minute)...")
    ml_results = run_ml(train_clean, val_clean)
    with open(OUT_ML_JSON, "w", encoding="utf-8") as f:
        json.dump(ml_results, f, indent=2)
    print(f"Saved ML results -> {OUT_ML_JSON}  (best model: {ml_results['best_model']})")

    print("Building dashboard...")
    build_dashboard(stats, ml_results, OUT_DASHBOARD_HTML)
    print(f"Saved dashboard -> {OUT_DASHBOARD_HTML}")

    print("\nDone. Open twitter_sentiment_dashboard.html in a browser.")


if __name__ == "__main__":
    main()
