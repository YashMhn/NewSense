"""
main.py
-------
Stage 1: Manual pipeline run.
  Discover -> Scrape -> Save to SQLite (+ optional CSV)

Stage progression:
  Stage 1: main.py       (this file) -- manual run
  Stage 2: scheduler.py             -- daily automated runs
  Stage 3: dags/                    -- Airflow (Linux/WSL2 only)

All configuration lives in config.py.
"""

import csv
import os
from collections import Counter
from datetime import datetime

from config import (
    DATA_DIR, EXPORT_CSV, MAX_PER_SOURCE,
    RSS_FEEDS, SITEMAP_SITES,
)
from database import init_db, insert_articles, print_db_stats
from discoverer import discover_all
from punkt_tab_downloader import ensure_punkt
from scraper import scrape_articles


def export_csv(articles: list[dict]) -> str:
    """
    Exports scraped articles to a timestamped CSV file.
    Useful for quick inspection without querying the DB.
    """
    if not articles:
        return ""

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"news_{timestamp}.csv")

    fieldnames = [
        "title", "authors", "date", "summary", "text",
        "tags", "url", "source", "language", "extracted_by", "scraped_at",
    ]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)

    print(f"  CSV export    : {filepath}")
    return filepath


def display_preview(articles: list[dict], n: int = 5) -> None:
    """Prints a terminal preview of the first n scraped articles."""
    print(f"\n{'='*65}")
    print(f"  SCRAPED ARTICLES PREVIEW (top {n})")
    print(f"{'='*65}")

    for i, a in enumerate(articles[:n], 1):
        print(f"\n[{i}] {a['title']}")
        print(f"    Source      : {a['source']}")
        print(f"    Authors     : {a['authors']}")
        print(f"    Date        : {a['date']}")
        print(f"    Language    : {a['language']}")
        print(f"    Tags        : {str(a['tags'])[:80]}")
        print(f"    Extracted by: {a['extracted_by']}")
        print(f"    Summary     : {a['summary'][:100]}{'...' if len(a['summary']) > 100 else ''}")

    print(f"\n{'='*65}\n")


def print_extractor_stats(articles: list[dict]) -> None:
    """Shows a visual breakdown of which extractor handled each article."""
    counts = Counter(a["extracted_by"] for a in articles)
    print(f"\n  Extractor breakdown:")
    for extractor, count in counts.most_common():
        bar = "█" * count
        print(f"    {extractor:<15} {bar} ({count})")


if __name__ == "__main__":
    print("=" * 65)
    print("  NEWS SCRAPER PIPELINE")
    print("=" * 65)

    # Ensure NLTK punkt_tab is available
    ensure_punkt()

    # Step 0: Initialise database
    print("\n[STEP 0] INITIALISING DATABASE")
    init_db()

    # Step 1: Discover URLs
    print("\n[STEP 1] DISCOVERING ARTICLE URLs")
    urls = discover_all(
        rss_feeds=RSS_FEEDS,
        sitemap_sites=SITEMAP_SITES,
        max_per_source=MAX_PER_SOURCE,
    )

    if not urls:
        print("\nNo URLs discovered. Check config.py sources.")
        exit(1)

    # Step 2: Scrape articles
    print(f"\n[STEP 2] SCRAPING {len(urls)} ARTICLES")
    articles, failed_count = scrape_articles(urls)

    if not articles:
        print("No articles scraped. Check your connection and try again.")
        exit(1)

    # Step 3: Save to database
    print(f"\n[STEP 3] SAVING TO DATABASE")
    inserted, duplicates = insert_articles(articles)
    print(f"  Inserted  : {inserted} new articles")
    print(f"  Duplicates: {duplicates} already in DB (skipped)")

    if EXPORT_CSV:
        export_csv(articles)

    # Step 4: Results
    display_preview(articles)
    print_extractor_stats(articles)
    print_db_stats()

    print(f"\n{'='*65}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*65}")
    print(f"  URLs discovered : {len(urls)}")
    print(f"  Scraped OK      : {len(articles)}")
    print(f"  New in DB       : {inserted}")
    print(f"  Duplicates      : {duplicates}")
    print(f"  Failed URLs     : {failed_count}"
          + (f" -> see {DATA_DIR}/failed_urls.log" if failed_count else ""))
    print(f"  Success rate    : {len(articles)/len(urls)*100:.1f}%")
    print(f"{'='*65}")
    print("\nPipeline complete!")
    print("  Next: run scheduler.py for daily automated runs.")
