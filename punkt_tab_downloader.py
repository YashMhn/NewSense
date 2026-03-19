"""
punkt_tab_downloader.py
-----------------------
Ensures NLTK's punkt_tab tokenizer data is available before
any NLP operations run. Downloads it only if not already present.

Called at startup in main.py, scheduler.py, and dashboard.py.
"""

import nltk


def ensure_punkt() -> None:
    """Downloads punkt_tab if not already present. No-op if already installed."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading NLTK punkt_tab...")
        nltk.download("punkt_tab", quiet=True)
        print("Done.")
