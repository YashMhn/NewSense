"""
sentiment.py
------------
Sentiment analysis engine for scraped news articles.
Runs both VADER and TextBlob on each article and stores
results back into the database (idempotent — never re-scores).

VADER    — rule-based, tuned for news/social text, fast, no training needed.
TextBlob — lexicon-based, simpler API, good general baseline.

Both scores are normalised to [-1.0, +1.0]:
  > +0.05  → positive
  < -0.05  → negative
   else    → neutral
"""

import os
import re
import sqlite3

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import DATA_DIR, DB_FILENAME
from database import DB_PATH, get_connection

# Initialised once at module level — cheap after first import
_vader = SentimentIntensityAnalyzer()

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "these", "those", "he", "she", "they",
    "we", "you", "i", "my", "his", "her", "their", "our", "your",
    "as", "if", "not", "no", "so", "up", "out", "about", "also", "than",
    "after", "before", "said", "says", "say", "new", "one", "two", "year",
    "years", "per", "cent", "news", "report", "reports", "day", "time",
    "how", "who", "what", "when", "why", "where", "more", "less", "all",
    "just", "over", "into", "while", "amid", "under", "says",
}


# ── Label helper ──────────────────────────────────────────────────────────────

def _label(score: float) -> str:
    """Converts a [-1, 1] compound score to a human-readable label."""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"


# ── Scoring ───────────────────────────────────────────────────────────────────

def _vader_score(text: str) -> dict:
    """Returns VADER sentiment scores for a piece of text."""
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    return {
        "vader_compound": round(compound, 4),
        "vader_label":    _label(compound),
        "vader_pos":      round(scores["pos"], 4),
        "vader_neu":      round(scores["neu"], 4),
        "vader_neg":      round(scores["neg"], 4),
    }


def _textblob_score(text: str) -> dict:
    """Returns TextBlob sentiment scores for a piece of text."""
    analysis = TextBlob(text)
    polarity = round(analysis.sentiment.polarity, 4)
    return {
        "tb_compound":     polarity,
        "tb_label":        _label(polarity),
        "tb_subjectivity": round(analysis.sentiment.subjectivity, 4),
    }


def score_text(title: str, text: str) -> dict:
    """
    Scores an article using both VADER and TextBlob.
    Uses title + first 500 chars of body — title carries strong signal,
    body adds context without making scoring slow.

    Returns a flat dict with all sentiment fields merged.
    """
    snippet = f"{title}. {text[:500]}" if text and text != "N/A" else title
    return {**_vader_score(snippet), **_textblob_score(snippet)}


def agreement_rate(df: pd.DataFrame) -> float:
    """
    Returns the % of articles where VADER and TextBlob produce
    the same sentiment label. A measure of scorer consistency.
    """
    if df.empty:
        return 0.0
    agreed = (df["vader_label"] == df["tb_label"]).sum()
    return round(agreed / len(df) * 100, 1)


# ── Database helpers ──────────────────────────────────────────────────────────

def _ensure_sentiment_columns(conn: sqlite3.Connection) -> None:
    """
    Lazily adds sentiment columns to the articles table.
    Safe to call on every run — silently skips existing columns.
    """
    columns = [
        "vader_compound REAL", "vader_label TEXT",
        "vader_pos REAL",      "vader_neu REAL",   "vader_neg REAL",
        "tb_compound REAL",    "tb_label TEXT",    "tb_subjectivity REAL",
    ]
    for col in columns:
        try:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # Column already exists


def score_database(db_path: str = DB_PATH) -> int:
    """
    Scores all unscored articles and writes results back to the DB.
    Only processes rows where vader_compound IS NULL — fully idempotent.

    Returns:
        Number of articles scored in this call.
    """
    conn = get_connection(db_path)
    _ensure_sentiment_columns(conn)

    rows = conn.execute("""
        SELECT id, title, text FROM articles
        WHERE vader_compound IS NULL
    """).fetchall()

    if not rows:
        conn.close()
        return 0

    update_sql = """
        UPDATE articles SET
            vader_compound   = :vader_compound,
            vader_label      = :vader_label,
            vader_pos        = :vader_pos,
            vader_neu        = :vader_neu,
            vader_neg        = :vader_neg,
            tb_compound      = :tb_compound,
            tb_label         = :tb_label,
            tb_subjectivity  = :tb_subjectivity
        WHERE id = :id
    """

    with conn:
        for row in rows:
            scores = score_text(row["title"] or "", row["text"] or "")
            conn.execute(update_sql, {"id": row["id"], **scores})

    conn.close()
    return len(rows)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_articles(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Loads all scored articles from the database as a DataFrame.
    Scores any unscored articles first (idempotent).

    Returns:
        DataFrame with all article + sentiment columns.
        Empty DataFrame if no articles exist.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = get_connection(db_path)
    _ensure_sentiment_columns(conn)
    conn.close()

    score_database(db_path)

    conn = get_connection(db_path)
    df = pd.read_sql_query("""
        SELECT
            id, title, url, source, date, authors,
            summary, scraped_at,
            vader_compound, vader_label, vader_pos, vader_neu, vader_neg,
            tb_compound, tb_label, tb_subjectivity
        FROM articles
        WHERE vader_compound IS NOT NULL
        ORDER BY scraped_at DESC
    """, conn)
    conn.close()

    df = df.fillna("N/A")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


# ── Word cloud helpers ────────────────────────────────────────────────────────

def get_wordcloud_text(
    df: pd.DataFrame,
    label: str,
    scorer: str = "vader",
) -> str:
    """
    Returns a cleaned string of all title words for articles
    matching the given sentiment label, filtered by chosen scorer.

    Args:
        df:     Full articles DataFrame
        label:  'positive', 'negative', or 'neutral'
        scorer: 'vader' or 'textblob' — which label column to filter on

    Returns:
        Space-joined string of words for the word cloud generator.
    """
    label_col = "vader_label" if scorer == "vader" else "tb_label"
    subset = df[df[label_col] == label]["title"].dropna()
    words = []
    for title in subset:
        tokens = re.findall(r"[a-zA-Z]{3,}", title.lower())
        words.extend(w for w in tokens if w not in STOPWORDS)
    return " ".join(words)


# ── Top headlines ─────────────────────────────────────────────────────────────

def get_top_headlines(
    df: pd.DataFrame,
    label: str,
    n: int = 10,
    scorer: str = "vader",
) -> pd.DataFrame:
    """
    Returns the top n most positive or negative headlines.

    Args:
        df:     Full articles DataFrame
        label:  'positive' or 'negative'
        n:      Number of headlines to return
        scorer: 'vader' or 'textblob'

    Returns:
        DataFrame with title, source, url, date, score columns.
    """
    col       = "vader_compound" if scorer == "vader" else "tb_compound"
    label_col = "vader_label"    if scorer == "vader" else "tb_label"

    subset    = df[df[label_col] == label].copy()
    ascending = label == "negative"

    return (
        subset[["title", "source", "url", "date", col]]
        .sort_values(col, ascending=ascending)
        .head(n)
        .rename(columns={col: "score"})
        .reset_index(drop=True)
    )
