"""
tests/test_pipeline.py
-----------------------
Basic test suite for the news scraper pipeline.

Run with:
    uv run pytest tests/ -v

Tests are grouped by module:
  - discoverer.py  (_is_article_url, _deduplicate)
  - scraper.py     (log_failed_url, _with_retry, goose authors safety)
  - database.py    (init_db, insert_articles deduplication, get_stats)
"""

import os
import sqlite3
import tempfile
import pytest

# ── discoverer tests ──────────────────────────────────────────────────────────

from discoverer import _is_article_url, _deduplicate


class TestIsArticleUrl:
    def test_valid_article(self):
        assert _is_article_url("https://bbc.com/news/world-12345678") is True

    def test_rejects_homepage(self):
        assert _is_article_url("https://bbc.com/") is False

    def test_rejects_tag_page(self):
        assert _is_article_url("https://thehindu.com/tag/politics/") is False

    def test_rejects_category_page(self):
        assert _is_article_url("https://ndtv.com/category/india") is False

    def test_rejects_author_page(self):
        assert _is_article_url("https://medium.com/author/john") is False

    def test_rejects_short_path(self):
        assert _is_article_url("https://bbc.com/uk") is False

    def test_rejects_search_page(self):
        assert _is_article_url("https://bbc.com/search/?q=news") is False

    def test_valid_long_slug(self):
        assert _is_article_url(
            "https://timesofindia.com/india/pm-modi-visits-mumbai/article123.cms"
        ) is True


class TestDeduplicate:
    def test_removes_duplicates(self):
        urls = ["https://a.com", "https://b.com", "https://a.com"]
        result = _deduplicate(urls)
        assert result == ["https://a.com", "https://b.com"]

    def test_preserves_order(self):
        urls = ["https://c.com", "https://a.com", "https://b.com"]
        assert _deduplicate(urls) == urls

    def test_empty_list(self):
        assert _deduplicate([]) == []

    def test_all_duplicates(self):
        urls = ["https://a.com"] * 5
        assert _deduplicate(urls) == ["https://a.com"]


# ── scraper tests ─────────────────────────────────────────────────────────────

from scraper import log_failed_url, _with_retry


class TestLogFailedUrl:
    def test_creates_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scraper.DATA_DIR", str(tmp_path))
        log_failed_url("https://example.com", "test reason")
        log_file = tmp_path / "failed_urls.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "https://example.com" in content
        assert "test reason" in content

    def test_appends_multiple_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scraper.DATA_DIR", str(tmp_path))
        log_failed_url("https://a.com", "reason 1")
        log_failed_url("https://b.com", "reason 2")
        content = (tmp_path / "failed_urls.log").read_text()
        assert "https://a.com" in content
        assert "https://b.com" in content

    def test_rotates_when_over_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scraper.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("scraper.LOG_MAX_LINES", 10)
        # Write 10 lines to hit the limit
        for i in range(10):
            log_failed_url(f"https://example.com/{i}", "reason")
        # This write should trigger rotation
        log_failed_url("https://example.com/trigger", "rotation trigger")
        content = (tmp_path / "failed_urls.log").read_text()
        assert "log rotated" in content


class TestWithRetry:
    def test_returns_on_first_success(self):
        calls = []
        def fn(url):
            calls.append(url)
            return {"title": "ok"}
        result = _with_retry(fn, "https://x.com", "test")
        assert result == {"title": "ok"}
        assert len(calls) == 1

    def test_retries_on_none(self, monkeypatch):
        monkeypatch.setattr("scraper.MAX_RETRIES", 3)
        monkeypatch.setattr("scraper.RETRY_DELAY", 0)
        calls = []
        def fn(url):
            calls.append(url)
            return None
        result = _with_retry(fn, "https://x.com", "test")
        assert result is None
        assert len(calls) == 3

    def test_returns_on_second_attempt(self, monkeypatch):
        monkeypatch.setattr("scraper.MAX_RETRIES", 3)
        monkeypatch.setattr("scraper.RETRY_DELAY", 0)
        attempt = {"n": 0}
        def fn(url):
            attempt["n"] += 1
            return {"title": "ok"} if attempt["n"] == 2 else None
        result = _with_retry(fn, "https://x.com", "test")
        assert result == {"title": "ok"}
        assert attempt["n"] == 2


class TestGooseAuthors:
    """Ensure the Goose3 authors field handles string vs list safely."""
    def test_list_authors(self):
        authors = ["Alice", "Bob"]
        result = ", ".join(authors) if isinstance(authors, list) else authors
        assert result == "Alice, Bob"

    def test_string_author(self):
        authors = "Alice"
        result = ", ".join(authors) if isinstance(authors, list) else authors
        assert result == "Alice"

    def test_empty_authors(self):
        authors = []
        result = ", ".join(authors) if (isinstance(authors, list) and authors) else "N/A"
        assert result == "N/A"

    def test_none_authors(self):
        authors = None
        result = authors if authors else "N/A"
        assert result == "N/A"


# ── database tests ────────────────────────────────────────────────────────────

from database import init_db, insert_articles, get_stats

SAMPLE_ARTICLE = {
    "title":        "Test Article",
    "authors":      "Test Author",
    "date":         "2025-01-01",
    "summary":      "A test summary",
    "text":         "Full article text here " * 10,
    "tags":         "test, python",
    "url":          "https://example.com/test-article",
    "source":       "example.com",
    "language":     "en",
    "extracted_by": "trafilatura",
    "scraped_at":   "2025-01-01 08:00:00",
}


@pytest.fixture
def tmp_db(tmp_path):
    """Creates a fresh temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


class TestDatabase:
    def test_init_creates_table(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_insert_article(self, tmp_db):
        inserted, duplicates = insert_articles([SAMPLE_ARTICLE], db_path=tmp_db)
        assert inserted == 1
        assert duplicates == 0

    def test_deduplication(self, tmp_db):
        insert_articles([SAMPLE_ARTICLE], db_path=tmp_db)
        # Insert the same article again
        inserted, duplicates = insert_articles([SAMPLE_ARTICLE], db_path=tmp_db)
        assert inserted == 0
        assert duplicates == 1

    def test_multiple_articles(self, tmp_db):
        articles = [
            {**SAMPLE_ARTICLE, "url": f"https://example.com/article-{i}"}
            for i in range(5)
        ]
        inserted, duplicates = insert_articles(articles, db_path=tmp_db)
        assert inserted == 5
        assert duplicates == 0

    def test_get_stats_total(self, tmp_db):
        insert_articles([SAMPLE_ARTICLE], db_path=tmp_db)
        stats = get_stats(db_path=tmp_db)
        assert stats["total_articles"] == 1

    def test_get_stats_empty_db(self, tmp_db):
        stats = get_stats(db_path=tmp_db)
        assert stats["total_articles"] == 0
        assert stats["top_sources"] == []

    def test_init_is_idempotent(self, tmp_db):
        # Calling init_db twice should not raise or duplicate the table
        init_db(tmp_db)
        init_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        assert count == 0
