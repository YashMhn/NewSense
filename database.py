"""
database.py
-----------
Handles all SQLite operations for the news scraper pipeline.

Responsibilities:
  - Create the articles table (if it doesn't exist)
  - Insert articles with URL-based deduplication (same URL is never stored twice)
  - Query stats for the pipeline summary

SQLite is used here because:
  - Zero setup — no server, no credentials, just a file
  - Built into Python (no install needed)
  - Easy to upgrade to PostgreSQL later by swapping the connection string
"""

import sqlite3
import os
from datetime import datetime


# Default database file path
DB_PATH = os.path.join("data", "news.db")


# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT,
    authors      TEXT,
    date         TEXT,
    summary      TEXT,
    text         TEXT,
    tags         TEXT,
    url          TEXT UNIQUE,       -- UNIQUE prevents duplicate articles
    source       TEXT,
    language     TEXT,
    extracted_by TEXT,
    scraped_at   TEXT,
    created_at   TEXT DEFAULT (datetime('now'))
);
"""

# Index on source and date for fast filtering/querying later
CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_source    ON articles (source);",
    "CREATE INDEX IF NOT EXISTS idx_date      ON articles (date);",
    "CREATE INDEX IF NOT EXISTS idx_language  ON articles (language);",
    "CREATE INDEX IF NOT EXISTS idx_scraped_at ON articles (scraped_at);",
]


# ── Connection helper ─────────────────────────────────────────────────────────

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Returns a SQLite connection with Row factory enabled
    (so rows can be accessed like dicts).
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    """
    Creates the database and articles table if they don't already exist.
    Safe to call every pipeline run — idempotent.

    Args:
        db_path: Path to the SQLite database file
    """
    conn = get_connection(db_path)
    with conn:
        conn.execute(CREATE_TABLE_SQL)
        for idx_sql in CREATE_INDEXES_SQL:
            conn.execute(idx_sql)
    conn.close()
    print(f"  Database ready: {db_path}")


# ── Insert ────────────────────────────────────────────────────────────────────

def insert_articles(
    articles: list[dict],
    db_path: str = DB_PATH
) -> tuple[int, int]:
    """
    Inserts a list of article dicts into the database.

    Uses INSERT OR IGNORE so articles with a URL already in the DB
    are silently skipped — this is how deduplication works across runs.

    Args:
        articles: List of article dicts from scraper.py
        db_path:  Path to the SQLite database file

    Returns:
        Tuple of (inserted_count, duplicate_count)
    """
    if not articles:
        return 0, 0

    insert_sql = """
        INSERT OR IGNORE INTO articles
            (title, authors, date, summary, text, tags, url,
             source, language, extracted_by, scraped_at)
        VALUES
            (:title, :authors, :date, :summary, :text, :tags, :url,
             :source, :language, :extracted_by, :scraped_at)
    """

    conn = get_connection(db_path)
    inserted = 0
    duplicates = 0

    with conn:
        for article in articles:
            cursor = conn.execute(insert_sql, article)
            if cursor.rowcount == 1:
                inserted += 1
            else:
                duplicates += 1

    conn.close()
    return inserted, duplicates


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats(db_path: str = DB_PATH) -> dict:
    """
    Returns summary statistics about the database.
    Used in the pipeline summary at the end of each run.

    Returns:
        Dict with total_articles, articles_today, top_sources, top_languages
    """
    conn = get_connection(db_path)
    today = datetime.now().strftime("%Y-%m-%d")

    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    today_count = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE scraped_at LIKE ?",
        (f"{today}%",)
    ).fetchone()[0]

    top_sources = conn.execute("""
        SELECT source, COUNT(*) as count
        FROM articles
        WHERE source != 'N/A'
        GROUP BY source
        ORDER BY count DESC
        LIMIT 5
    """).fetchall()

    top_languages = conn.execute("""
        SELECT language, COUNT(*) as count
        FROM articles
        WHERE language != 'N/A'
        GROUP BY language
        ORDER BY count DESC
        LIMIT 5
    """).fetchall()

    conn.close()

    return {
        "total_articles":  total,
        "articles_today":  today_count,
        "top_sources":     [(r["source"], r["count"]) for r in top_sources],
        "top_languages":   [(r["language"], r["count"]) for r in top_languages],
    }


def print_db_stats(db_path: str = DB_PATH) -> None:
    """Prints a formatted database stats summary to the terminal."""
    stats = get_stats(db_path)

    print(f"\n  Database stats ({db_path}):")
    print(f"    Total articles : {stats['total_articles']}")
    print(f"    Added today    : {stats['articles_today']}")

    if stats["top_sources"]:
        print(f"\n    Top sources:")
        for source, count in stats["top_sources"]:
            bar = "█" * min(count, 20)
            print(f"      {source:<30} {bar} ({count})")

    if stats["top_languages"]:
        print(f"\n    Top languages:")
        for lang, count in stats["top_languages"]:
            print(f"      {lang:<10} ({count})")
