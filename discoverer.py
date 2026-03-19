"""
discoverer.py
-------------
Discovers article URLs from three sources:
  1. RSS Feeds     — latest articles from any news site
  2. Google News   — search articles by keyword
  3. Sitemap       — crawl a site's sitemap.xml for article URLs

All three return a flat list of URLs, ready to pass into scraper.py.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from urllib.parse import urlparse


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_article_url(url: str) -> bool:
    """
    Basic heuristic to filter out non-article URLs from sitemaps.
    Skips homepages, tag pages, category pages, etc.
    """
    skip_patterns = [
        "/tag/", "/tags/", "/category/", "/author/",
        "/page/", "/feed/", "/search/", "/?", "/about",
        "/contact", "/privacy", "/terms"
    ]
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Must have a meaningful path (not just "/")
    if len(path) < 5:
        return False

    # Skip known non-article patterns
    if any(p in path for p in skip_patterns):
        return False

    return True


def _deduplicate(urls: list[str]) -> list[str]:
    """Removes duplicate URLs while preserving order."""
    seen = set()
    result = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


# ── Source 1: RSS Feeds ───────────────────────────────────────────────────────

# Common RSS feed paths to try if none is specified
_RSS_SUFFIXES = [
    "/feed", "/feed/", "/rss", "/rss/", "/rss.xml",
    "/feed.xml", "/atom.xml", "/news/rss", "/feeds/posts/default"
]

def discover_from_rss(feed_urls: list[str], max_per_feed: int = 10) -> list[str]:
    """
    Pulls latest article URLs from a list of RSS/Atom feed URLs.

    Args:
        feed_urls:    List of RSS feed URLs
                      (e.g. "https://feeds.bbci.co.uk/news/rss.xml")
        max_per_feed: Max articles to pull from each feed

    Returns:
        List of article URLs.
    """
    urls = []

    for feed_url in feed_urls:
        print(f"  📡 RSS: {feed_url}")
        try:
            feed = feedparser.parse(feed_url)

            # Distinguish a fetch failure from a genuinely empty feed
            if feed.bozo and not feed.entries:
                print(f"     ⚠️  Feed failed to load: {feed.bozo_exception}")
                continue

            if not feed.entries:
                print(f"     ⚠️  Feed loaded but has no entries")
                continue

            count = 0
            for entry in feed.entries[:max_per_feed]:
                link = entry.get("link")
                if link:
                    urls.append(link)
                    count += 1

            print(f"     ✅ {count} URLs found")

        except Exception as e:
            print(f"     ❌ Failed: {e}")

    return urls


def find_rss_feed(site_url: str) -> str | None:
    """
    Tries to auto-discover an RSS feed URL for a given site.
    Useful when you only have the homepage URL.

    Args:
        site_url: Homepage URL (e.g. "https://timesofindia.com")

    Returns:
        RSS feed URL if found, else None.
    """
    headers = {"User-Agent": "Mozilla/5.0"}

    # Try common RSS suffixes
    base = site_url.rstrip("/")
    for suffix in _RSS_SUFFIXES:
        try:
            url = base + suffix
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200 and "xml" in resp.headers.get("Content-Type", ""):
                print(f"  🎯 Found RSS at: {url}")
                return url
        except Exception:
            continue

    # Try to find <link rel="alternate"> in the homepage HTML
    try:
        resp = requests.get(site_url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        rss_tag = soup.find("link", rel="alternate", type="application/rss+xml")
        if rss_tag and rss_tag.get("href"):
            href = rss_tag["href"]
            if href.startswith("/"):
                href = base + href
            print(f"  🎯 Found RSS in HTML: {href}")
            return href
    except Exception:
        pass

    print(f"  ⚠️  No RSS feed found for {site_url}")
    return None

# ── Source 2: Sitemap Crawling ────────────────────────────────────────────────

def discover_from_sitemap(site_url: str, max_urls: int = 30) -> list[str]:
    """
    Crawls a site's sitemap.xml to find article URLs.
    Handles nested sitemaps (sitemap index files) automatically.

    Args:
        site_url: Homepage or sitemap URL
                  (e.g. "https://thehindu.com" or "https://thehindu.com/sitemap.xml")
        max_urls: Max article URLs to return

    Returns:
        List of article URLs.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    base = site_url.rstrip("/")

    # Guess sitemap URL if homepage given
    sitemap_url = site_url if site_url.endswith(".xml") else base + "/sitemap.xml"
    print(f"  🗺️  Sitemap: {sitemap_url}")

    urls = []

    def parse_sitemap(url: str, depth: int = 0):
        """Recursively parses sitemap and nested sitemaps."""
        if len(urls) >= max_urls or depth > 2:
            return
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(resp.content, "xml")

            # Sitemap Index — contains links to other sitemaps
            nested = soup.find_all("sitemap")
            if nested:
                for s in nested[:5]:   # Limit nested sitemaps explored
                    loc = s.find("loc")
                    if loc:
                        parse_sitemap(loc.text.strip(), depth + 1)
                return

            # Regular sitemap — contains article URLs
            locs = soup.find_all("loc")
            for loc in locs:
                if len(urls) >= max_urls:
                    break
                url_text = loc.text.strip()
                if _is_article_url(url_text):
                    urls.append(url_text)

        except Exception as e:
            print(f"     ⚠️  Sitemap parse error: {e}")

    parse_sitemap(sitemap_url)
    print(f"     ✅ {len(urls)} article URLs found")
    return urls


# ── Combined Discoverer ───────────────────────────────────────────────────────

def discover_all(
    rss_feeds: list[str] = None,
    sitemap_sites: list[str] = None,
    max_per_source: int = 10,
) -> list[str]:
    """
    Discovers article URLs from RSS feeds and sitemaps, returning
    a deduplicated combined list ready for scraping.

    Args:
        rss_feeds:      List of RSS feed URLs
        sitemap_sites:  List of site URLs to crawl sitemaps for
        max_per_source: Max articles per individual feed/site

    Returns:
        Deduplicated list of article URLs.
    """
    all_urls = []

    if rss_feeds:
        print("\n📡 [1/2] Discovering via RSS Feeds...")
        all_urls += discover_from_rss(rss_feeds, max_per_feed=max_per_source)

    if sitemap_sites:
        print("\n🗺️  [2/2] Discovering via Sitemaps...")
        for site in sitemap_sites:
            all_urls += discover_from_sitemap(site, max_urls=max_per_source)

    deduped = _deduplicate(all_urls)
    print(f"\n🔗 Total unique URLs discovered: {len(deduped)} "
          f"(from {len(all_urls)} raw, {len(all_urls) - len(deduped)} duplicates removed)")
    return deduped
