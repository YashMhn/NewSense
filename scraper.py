"""
scraper.py
----------
Scrapes news articles using a three-layer extraction strategy:
  1. Trafilatura  -- fastest, highest accuracy (F1: 0.958)
  2. Newspaper4k  -- fallback with NLP extras (keywords, summary)
  3. Goose3       -- fallback, great for Asian/Indian news sites

Improvements over naive scraping:
  - Per-extractor retry logic (transient network blips)
  - ThreadPoolExecutor for concurrent scraping (~5x faster)
  - Rotating failed_urls.log (never grows unbounded)
  - Verbose failure messages at every silent failure point
  - Goose3 authors type safety (string vs list)
  - Newspaper4k request timeout to prevent hangs
"""

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

import trafilatura
from trafilatura.settings import use_config
from goose3 import Goose
from newspaper import Article as NewspaperArticle

from config import (
    DATA_DIR, LOG_MAX_LINES, MAX_RETRIES, MAX_WORKERS,
    MIN_TEXT_LENGTH, REQUEST_DELAY, RETRY_DELAY,
)


# ── Extractor setup ───────────────────────────────────────────────────────────

_traf_config = use_config()
_traf_config.set("DEFAULT", "DOWNLOAD_TIMEOUT", "10")

# Goose3 is NOT thread-safe -- create one instance per call via factory
def _make_goose() -> Goose:
    return Goose({
        "browser_user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        ),
        "http_timeout": 10,
    })

# Lock for writing to the log file from multiple threads
_log_lock = Lock()


# ── Failed URL Logger ─────────────────────────────────────────────────────────

def log_failed_url(url: str, reason: str) -> None:
    """
    Appends a failed URL with reason and timestamp to failed_urls.log.
    Rotates the log if it exceeds LOG_MAX_LINES to prevent unbounded growth.
    Thread-safe via a lock.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    log_path = os.path.join(DATA_DIR, "failed_urls.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {reason} | {url}\n"

    with _log_lock:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) >= LOG_MAX_LINES:
                rotated = lines[LOG_MAX_LINES // 2:]
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] --- log rotated ---\n")
                    f.writelines(rotated)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, url: str, label: str) -> dict | None:
    """
    Calls fn(url) up to MAX_RETRIES times.
    Returns the result on first success, None if all attempts fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        result = fn(url)
        if result:
            return result
        if attempt < MAX_RETRIES:
            print(f"   ↩  {label}: retry {attempt}/{MAX_RETRIES - 1}...")
            time.sleep(RETRY_DELAY)
    return None


# ── Layer 1: Trafilatura ──────────────────────────────────────────────────────

def _scrape_with_trafilatura(url: str) -> dict | None:
    """Primary extraction using Trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print(f"   ⚠️  Trafilatura: fetch returned nothing (rate limited or blocked)")
            return None

        result = trafilatura.extract(
            downloaded,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            favor_precision=True,
        )

        if not result:
            print(f"   ⚠️  Trafilatura: page fetched but no content extracted")
            return None

        data = json.loads(result)
        text = data.get("text") or ""

        if len(text) < MIN_TEXT_LENGTH:
            print(f"   ⚠️  Trafilatura: text too short ({len(text)} chars)")
            return None

        return {
            "title":        data.get("title") or "N/A",
            "authors":      data.get("author") or "N/A",
            "date":         data.get("date") or "N/A",
            "summary":      data.get("description") or "N/A",
            "text":         text,
            "tags":         data.get("tags") or "N/A",
            "url":          url,
            "source":       data.get("sitename") or "N/A",
            "language":     data.get("language") or "N/A",
            "extracted_by": "trafilatura",
            "scraped_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        print(f"   ⚠️  Trafilatura error: {e}")
        return None


# ── Layer 2: Newspaper4k ──────────────────────────────────────────────────────

def _scrape_with_newspaper(url: str) -> dict | None:
    """Fallback extraction using Newspaper4k. Adds NLP keywords + summary."""
    try:
        article = NewspaperArticle(url, request_timeout=10)
        article.download()
        article.parse()
        article.nlp()

        text = article.text or ""
        if not article.title or len(text) < MIN_TEXT_LENGTH:
            print(f"   ⚠️  Newspaper4k: text too short or no title")
            return None

        return {
            "title":        article.title,
            "authors":      ", ".join(article.authors) if article.authors else "N/A",
            "date":         str(article.publish_date.date()) if article.publish_date else "N/A",
            "summary":      article.summary or "N/A",
            "text":         text,
            "tags":         ", ".join(article.keywords) if article.keywords else "N/A",
            "url":          article.url or url,
            "source":       article.source_url or "N/A",
            "language":     "N/A",
            "extracted_by": "newspaper4k",
            "scraped_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        print(f"   ⚠️  Newspaper4k error: {e}")
        return None


# ── Layer 3: Goose3 ───────────────────────────────────────────────────────────

def _scrape_with_goose(url: str) -> dict | None:
    """
    Second fallback using Goose3.
    Effective on Indian/Asian news sites (TOI, The Hindu, NDTV).
    Creates a fresh Goose instance per call -- Goose3 is not thread-safe.
    """
    try:
        goose = _make_goose()
        article = goose.extract(url=url)

        text = article.cleaned_text or ""
        if not article.title or len(text) < MIN_TEXT_LENGTH:
            print(f"   ⚠️  Goose3: text too short or no title")
            return None

        date = "N/A"
        if article.publish_date:
            date = str(article.publish_date)[:10]

        tags = ", ".join(article.tags) if article.tags else "N/A"

        # authors can be a string or a list depending on the site
        authors = article.authors
        if isinstance(authors, list):
            authors = ", ".join(authors) if authors else "N/A"
        elif not authors:
            authors = "N/A"

        return {
            "title":        article.title,
            "authors":      authors,
            "date":         date,
            "summary":      article.meta_description or "N/A",
            "text":         text,
            "tags":         tags,
            "url":          article.final_url or url,
            "source":       article.domain or "N/A",
            "language":     article.meta_lang or "N/A",
            "extracted_by": "goose3",
            "scraped_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        print(f"   ⚠️  Goose3 error: {e}")
        return None


# ── Single article pipeline ───────────────────────────────────────────────────

def scrape_article(url: str) -> dict | None:
    """
    Scrapes one article through the three-layer fallback chain, with retries.
    Logs to failed_urls.log if all layers fail.

    Chain: Trafilatura -> Newspaper4k -> Goose3 -> log & skip
    """
    result = _with_retry(_scrape_with_trafilatura, url, "Trafilatura")
    if result:
        print(f"   ✅ Trafilatura")
        return result

    print(f"   🔄 Trying Newspaper4k...")
    result = _with_retry(_scrape_with_newspaper, url, "Newspaper4k")
    if result:
        print(f"   ✅ Newspaper4k")
        return result

    print(f"   🔄 Trying Goose3...")
    result = _with_retry(_scrape_with_goose, url, "Goose3")
    if result:
        print(f"   ✅ Goose3")
        return result

    print(f"   ❌ All extractors failed.")
    log_failed_url(url, reason="all extractors failed")
    return None


# ── Concurrent batch scraper ──────────────────────────────────────────────────

def scrape_articles(urls: list[str]) -> tuple[list[dict], int]:
    """
    Scrapes a list of URLs concurrently using ThreadPoolExecutor.

    Uses MAX_WORKERS threads with a staggered submission to avoid
    burst requests. Each thread runs the full fallback chain independently.

    Args:
        urls: List of direct article URLs

    Returns:
        Tuple of (successfully scraped articles, count of failures)
    """
    results = []
    failed = 0
    total = len(urls)

    print(f"  Scraping {total} URLs with {MAX_WORKERS} workers...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {}
        for i, url in enumerate(urls):
            future = executor.submit(scrape_article, url)
            future_to_url[future] = (i + 1, url)
            # Stagger submissions to avoid bursting all workers at once
            if i < total - 1:
                time.sleep(REQUEST_DELAY / MAX_WORKERS)

        for future in as_completed(future_to_url):
            idx, url = future_to_url[future]
            print(f"\n🔍 [{idx}/{total}] {url}")
            try:
                article = future.result()
                if article:
                    results.append(article)
                    print(f"   📰 '{article['title'][:65]}'")
                else:
                    failed += 1
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                traceback.print_exc()
                log_failed_url(url, reason=f"unexpected error: {e}")
                failed += 1

    print(f"\n📦 Scraped: {len(results)} success, {failed} failed")
    if failed:
        print(f"   📋 Failed URLs logged to: {DATA_DIR}/failed_urls.log")

    return results, failed
