"""
scheduler.py
------------
Runs the news scraper pipeline on a daily schedule.
Uses the lightweight `schedule` library -- no external infrastructure needed.

Stage 2 of the pipeline progression:
  Stage 1: Manual run (main.py)
  Stage 2: Scheduled run (this file)
  Stage 3: Airflow DAG (dags/) -- Linux/WSL2 only

Usage:
    uv run python scheduler.py

Keep this running in the background (or in a tmux/screen session on a server).
"""

import schedule
import time
import traceback
from collections import Counter
from datetime import datetime

from config import (
    RSS_FEEDS, SITEMAP_SITES,
    MAX_PER_SOURCE, RUN_TIME, DATA_DIR,
)
from database import init_db, insert_articles, print_db_stats
from discoverer import discover_all
from punkt_tab_downloader import ensure_punkt
from scraper import scrape_articles


def run_pipeline() -> None:
    """
    Executes one full pipeline run:
      1. Discover URLs from all configured sources
      2. Scrape and extract articles (concurrent, with fallback chain)
      3. Insert into SQLite with deduplication
      4. Print run summary and DB stats
    """
    run_start = datetime.now()
    print(f"\n{'='*65}")
    print(f"  SCHEDULED RUN -- {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

    try:
        # Step 1: Discover
        print("\n[STEP 1] DISCOVERING ARTICLE URLs")
        urls = discover_all(
            rss_feeds=RSS_FEEDS,
            sitemap_sites=SITEMAP_SITES,
            max_per_source=MAX_PER_SOURCE,
        )

        if not urls:
            print("  No URLs discovered. Skipping this run.")
            return

        # Step 2: Scrape
        print(f"\n[STEP 2] SCRAPING {len(urls)} ARTICLES")
        articles, failed_count = scrape_articles(urls)

        if not articles:
            print("  No articles scraped. Check connection.")
            return

        # Step 3: Save to database
        print(f"\n[STEP 3] SAVING TO DATABASE")
        inserted, duplicates = insert_articles(articles)
        print(f"  Inserted  : {inserted} new articles")
        print(f"  Duplicates: {duplicates} already in DB (skipped)")

        # Extractor breakdown
        counts = Counter(a["extracted_by"] for a in articles)
        print(f"\n  Extractor breakdown:")
        for extractor, count in counts.most_common():
            print(f"    {'█' * count} {extractor} ({count})")

        # Step 4: Summary
        elapsed = (datetime.now() - run_start).seconds
        print(f"\n{'='*65}")
        print(f"  RUN SUMMARY")
        print(f"{'='*65}")
        print(f"  URLs discovered : {len(urls)}")
        print(f"  Scraped OK      : {len(articles)}")
        print(f"  New in DB       : {inserted}")
        print(f"  Duplicates      : {duplicates}")
        print(f"  Failed URLs     : {failed_count}"
              + (f" -> see {DATA_DIR}/failed_urls.log" if failed_count else ""))
        print(f"  Time taken      : {elapsed}s")

        print_db_stats()

    except Exception as e:
        print(f"\n  PIPELINE ERROR: {e}")
        traceback.print_exc()

    print(f"\n  Next run scheduled at {RUN_TIME} tomorrow.")


if __name__ == "__main__":
    print(f"  News Scraper Scheduler starting...")
    print(f"  Pipeline will run daily at {RUN_TIME}")
    print(f"  Press Ctrl+C to stop.\n")

    # Ensure NLTK data is available before first run
    ensure_punkt()

    # Initialise the database
    init_db()

    # Schedule the daily run
    schedule.every().day.at(RUN_TIME).do(run_pipeline)

    # Run immediately on startup
    print("  Running pipeline now (startup run)...")
    run_pipeline()

    # Keep the scheduler alive
    while True:
        schedule.run_pending()
        time.sleep(60)
