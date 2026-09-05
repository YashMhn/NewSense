"""
config.py
---------
Single source of truth for all pipeline configuration.
Edit this file to change sources, schedule, or behaviour.
Imported by main.py, scheduler.py, and the Airflow DAG.
"""

import os

# ── Discovery ─────────────────────────────────────────────────────────────────

# RSS feed URLs — add or remove any RSS/Atom feed
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://feeds.feedburner.com/ndtvnews-top-stories",
]

# Sites to discover articles from via sitemap.xml
SITEMAP_SITES = [
    "https://www.thehindu.com",
]

# Max articles to pull per individual feed / keyword / site
MAX_PER_SOURCE = 10

# ── Scraping ──────────────────────────────────────────────────────────────────

# Seconds to wait between requests — increase if getting rate limited
REQUEST_DELAY = 1.0

# Max concurrent threads for scraping (ThreadPoolExecutor)
MAX_WORKERS = 5

# Max retries per article before falling to next extractor layer
MAX_RETRIES = 2

# Seconds to wait between retries
RETRY_DELAY = 2.0

# Minimum article text length to be considered valid (chars)
MIN_TEXT_LENGTH = 100

# ── Storage ───────────────────────────────────────────────────────────────────

# Directory for all output files (DB, CSV, logs).
# Anchored to the project root so every entry point writes to the same place
# regardless of the working directory it was launched from — Airflow in
# particular runs tasks from its own cwd.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# SQLite database filename
DB_FILENAME = "news.db"

# Max lines in failed_urls.log before it gets rotated
LOG_MAX_LINES = 1000

# ── Scheduler ─────────────────────────────────────────────────────────────────

# Daily run time (24-hour format, local time)
RUN_TIME = "08:00"

# ── Export ────────────────────────────────────────────────────────────────────

# Set to True to also write a CSV alongside the database each run
EXPORT_CSV = True
