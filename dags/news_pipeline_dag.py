"""
dags/news_pipeline_dag.py
--------------------------
Apache Airflow DAG for the news scraper pipeline.

This is Stage 3 of the pipeline progression:
  Stage 1: Manual run    (main.py)
  Stage 2: Scheduled run (scheduler.py)
  Stage 3: Airflow DAG   (this file)  ← production-grade orchestration

Why Airflow over scheduler.py?
  - Each step is a separate task — failures are isolated and retryable
  - Full run history, logs, and alerting in the Airflow UI
  - Tasks can be parallelised (e.g. scrape multiple batches at once)
  - Industry standard for data pipeline orchestration

Setup:
    1. Install Airflow:  pip install apache-airflow
    2. Copy this file to your Airflow DAGs folder (usually ~/airflow/dags/)
    3. Start Airflow:    airflow standalone
    4. Open the UI:      http://localhost:8080
    5. Enable the DAG:   news_scraper_pipeline
"""

from __future__ import annotations

import sys
import os

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Make sure the project root is on the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discoverer import discover_all
from scraper import scrape_articles
from database import init_db, insert_articles


# ── DAG default args ──────────────────────────────────────────────────────────

default_args = {
    "owner":            "airflow",
    "depends_on_past":  False,
    "email_on_failure": False,     # Set to True and add your email in production
    "email_on_retry":   False,
    "retries":          2,         # Retry failed tasks twice before marking failed
    "retry_delay":      timedelta(minutes=5),
}

# ── Source config ─────────────────────────────────────────────────────────────

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://feeds.feedburner.com/ndtvnews-top-stories",
]

KEYWORDS = [
    "AI India 2025",
    "startup funding India",
    "technology news",
]

SITEMAP_SITES = [
    "https://www.thehindu.com",
]

MAX_PER_SOURCE = 10


# ── Task functions ────────────────────────────────────────────────────────────
# Each function below becomes one task in the Airflow UI.
# XCom (cross-communication) is used to pass data between tasks.

def task_init_db(**context) -> None:
    """Task 1: Ensure the database and schema exist."""
    init_db()
    print("Database initialised successfully.")


def task_discover_urls(**context) -> list[str]:
    """
    Task 2: Discover article URLs from all sources.
    Pushes the URL list to XCom for the next task to pick up.
    """
    urls = discover_all(
        rss_feeds=RSS_FEEDS,
        keywords=KEYWORDS,
        sitemap_sites=SITEMAP_SITES,
        max_per_source=MAX_PER_SOURCE,
    )

    if not urls:
        raise ValueError("No URLs discovered — check RSS feeds and keywords.")

    print(f"Discovered {len(urls)} unique URLs.")

    # Push to XCom so next task can access it
    context["ti"].xcom_push(key="urls", value=urls)
    return urls


def task_scrape_articles(**context) -> None:
    """
    Task 3: Scrape articles from discovered URLs using fallback chain.
    Pulls URL list from XCom, pushes scraped articles back to XCom.
    """
    # Pull URLs from previous task
    urls = context["ti"].xcom_pull(task_ids="discover_urls", key="urls")

    if not urls:
        raise ValueError("No URLs received from discover_urls task.")

    articles, failed_count = scrape_articles(urls)

    if not articles:
        raise ValueError("No articles scraped — check connection and URLs.")

    print(f"Scraped {len(articles)} articles. Failed: {failed_count}.")

    # Push articles to XCom for the save task
    context["ti"].xcom_push(key="articles", value=articles)
    context["ti"].xcom_push(key="failed_count", value=failed_count)


def task_save_to_db(**context) -> None:
    """
    Task 4: Save scraped articles to SQLite with deduplication.
    Pulls articles from XCom and inserts into the database.
    """
    articles = context["ti"].xcom_pull(task_ids="scrape_articles", key="articles")

    if not articles:
        raise ValueError("No articles received from scrape_articles task.")

    inserted, duplicates = insert_articles(articles)

    print(f"Inserted {inserted} new articles. Skipped {duplicates} duplicates.")

    # Log summary for the Airflow UI
    failed_count = context["ti"].xcom_pull(
        task_ids="scrape_articles", key="failed_count"
    ) or 0

    print(f"\nRun summary:")
    print(f"  Articles scraped : {len(articles)}")
    print(f"  New in DB        : {inserted}")
    print(f"  Duplicates       : {duplicates}")
    print(f"  Failed URLs      : {failed_count}")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="news_scraper_pipeline",
    description="Daily news scraper: discover → scrape → store in SQLite",
    default_args=default_args,
    schedule_interval="0 8 * * *",   # Every day at 08:00 (cron syntax)
    start_date=datetime(2025, 1, 1),
    catchup=False,                    # Don't backfill missed runs
    tags=["news", "scraper", "data-engineering"],
) as dag:

    # Task 1
    init_database = PythonOperator(
        task_id="init_database",
        python_callable=task_init_db,
    )

    # Task 2
    discover_urls = PythonOperator(
        task_id="discover_urls",
        python_callable=task_discover_urls,
    )

    # Task 3
    scrape_articles_task = PythonOperator(
        task_id="scrape_articles",
        python_callable=task_scrape_articles,
    )

    # Task 4
    save_to_db = PythonOperator(
        task_id="save_to_db",
        python_callable=task_save_to_db,
    )

    # ── Task dependencies (defines the execution order) ───────────────────────
    #
    #   init_database >> discover_urls >> scrape_articles >> save_to_db
    #
    init_database >> discover_urls >> scrape_articles_task >> save_to_db
