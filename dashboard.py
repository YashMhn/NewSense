"""
dashboard.py
------------
Streamlit sentiment dashboard for scraped news articles.

Sections:
  - KPI strip         : total articles, sources, scorer agreement rate
  - Sentiment donut   : VADER vs TextBlob distribution side by side
  - Scorer scatter    : per-article VADER vs TextBlob agreement plot
  - Source bar chart  : average sentiment per source, both scorers
  - Top headlines     : most positive / negative clickable headlines
  - Word clouds       : title words by sentiment, respects scorer toggle
  - Raw data table    : full scored dataset, sortable and filterable

Run with:
    uv run streamlit run dashboard.py

If data/news.db is missing or empty, fresh articles are scraped first.
"""

import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from config import MAX_PER_SOURCE, RSS_FEEDS, SITEMAP_SITES
from database import DB_PATH, init_db
from sentiment import (
    _label,
    agreement_rate,
    get_top_headlines,
    get_wordcloud_text,
    load_articles,
    score_database,
)

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="News Sentiment Dashboard",
    page_icon="📰",
    layout="wide",
)

LABEL_COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#95a5a6",
    "negative": "#e74c3c",
}

VADER_COLOR   = "#5b6ee1"
TEXTBLOB_COLOR = "#f0a500"


# ── Data pipeline ─────────────────────────────────────────────────────────────

def _scrape_fresh() -> bool:
    """
    Runs the discovery + scraper pipeline and inserts results into the DB.
    Returns True if at least one article was inserted.
    """
    from database import insert_articles
    from discoverer import discover_all
    from punkt_tab_downloader import ensure_punkt
    from scraper import scrape_articles

    ensure_punkt()
    init_db()

    urls = discover_all(
        rss_feeds=RSS_FEEDS,
        sitemap_sites=SITEMAP_SITES,
        max_per_source=MAX_PER_SOURCE,
    )
    if not urls:
        st.error("No URLs discovered — check config.py sources.")
        return False

    articles, _ = scrape_articles(urls)
    if not articles:
        st.error("Scraping returned no articles — check your connection.")
        return False

    insert_articles(articles)
    st.success(f"Scraped and stored {len(articles)} articles.")
    return True


@st.cache_data(ttl=300, show_spinner=False)
def get_data() -> pd.DataFrame:
    """
    Loads articles from the DB, scraping fresh if needed.
    Cached for 5 minutes — interactions don't re-hit the DB.
    Cache cleared by the refresh button.
    """
    db_missing = not os.path.exists(DB_PATH)
    db_empty   = False

    if not db_missing:
        conn  = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        db_empty = count == 0

    if db_missing or db_empty:
        st.info("No articles in database — running scraper now...")
        with st.spinner("Scraping articles (this takes ~1 minute)..."):
            ok = _scrape_fresh()
        if not ok:
            return pd.DataFrame()

    newly_scored = score_database()
    if newly_scored:
        st.toast(f"Scored {newly_scored} new articles.", icon="✅")

    return load_articles()


# ── Render helpers ────────────────────────────────────────────────────────────

def score_chip(score: float) -> str:
    """Coloured score badge string for markdown rendering."""
    if score > 0.05:
        return f"🟢 `{score:+.3f}`"
    elif score < -0.05:
        return f"🔴 `{score:+.3f}`"
    return f"⚪ `{score:+.3f}`"


def make_wordcloud(text: str, sentiment: str) -> plt.Figure:
    """Renders a word cloud figure. Transparent background for dark mode."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_alpha(0)

    if not text.strip():
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                fontsize=13, color="#888")
        ax.axis("off")
        return fig

    wc = WordCloud(
        width=800, height=350,
        background_color=None,
        mode="RGBA",
        colormap="Greens" if sentiment == "positive" else "Reds",
        max_words=80,
        collocations=False,
        prefer_horizontal=0.85,
    ).generate(text)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def render_headlines(headlines: pd.DataFrame) -> None:
    """Renders a list of headlines as clickable links with score chips."""
    if headlines.empty:
        st.info("Not enough articles for this category.")
        return
    for _, row in headlines.iterrows():
        title = row["title"]
        url   = row["url"]
        src   = row["source"]
        score = row["score"]
        st.markdown(
            f"**[{title}]({url})**  \n"
            f"<small style='color:gray'>{src}</small> &nbsp; {score_chip(score)}",
            unsafe_allow_html=True,
        )
        st.write("")  # Single blank line — cleaner than st.divider() every row


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📰 News Sentiment Dashboard")
    st.caption(
        "VADER vs TextBlob — comparing sentiment across your scraped news articles."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        scorer = st.radio(
            "Active scorer",
            ["vader", "textblob"],
            format_func=lambda x: "VADER" if x == "vader" else "TextBlob",
        )

        n_headlines = st.slider("Headlines to show", 5, 20, 10)

        st.divider()
        st.header("Filters")

        # Source filter — populated after data loads
        source_placeholder = st.empty()

        st.divider()
        if st.button("🔄 Refresh data", width='stretch'):
            st.cache_data.clear()
            st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading articles..."):
        df = get_data()

    if df.empty:
        st.warning("No data available. Check your connection and try refreshing.")
        st.stop()

    # Populate source filter now that data is loaded
    sources = sorted([s for s in df["source"].unique() if s != "N/A"])
    selected_sources = source_placeholder.multiselect(
        "Sources", sources, default=sources
    )

    filtered = df[df["source"].isin(selected_sources)] if selected_sources else df

    if filtered.empty:
        st.warning("No articles match the selected sources.")
        st.stop()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Articles", len(filtered))
    k2.metric("Sources", filtered["source"].nunique())

    pos_pct = round(
        (filtered["vader_label"] == "positive").sum() / len(filtered) * 100, 1
    )
    neg_pct = round(
        (filtered["vader_label"] == "negative").sum() / len(filtered) * 100, 1
    )
    k3.metric("Positive (VADER)", f"{pos_pct}%")
    k4.metric("Negative (VADER)", f"{neg_pct}%")

    agree = agreement_rate(filtered)
    k5.metric(
        "Scorer agreement",
        f"{agree}%",
        help="% of articles where VADER and TextBlob assign the same label",
    )

    st.divider()

    # ── Sentiment distribution ────────────────────────────────────────────────
    st.subheader("Sentiment distribution")

    d1, d2 = st.columns(2)

    with d1:
        vc = filtered["vader_label"].value_counts().reset_index()
        vc.columns = ["label", "count"]
        fig = px.pie(
            vc, values="count", names="label",
            title="VADER",
            color="label", color_discrete_map=LABEL_COLORS,
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig, width='stretch')

    with d2:
        tc = filtered["tb_label"].value_counts().reset_index()
        tc.columns = ["label", "count"]
        fig = px.pie(
            tc, values="count", names="label",
            title="TextBlob",
            color="label", color_discrete_map=LABEL_COLORS,
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig, width='stretch')

    # ── Scorer agreement scatter ───────────────────────────────────────────────
    st.subheader("Do VADER and TextBlob agree?")
    st.caption(
        "Each dot is one article. "
        "Points near the diagonal line mean both scorers agree. "
        "Outliers are worth investigating — the two methods disagree."
    )

    fig = px.scatter(
        filtered,
        x="vader_compound",
        y="tb_compound",
        color="vader_label",
        color_discrete_map=LABEL_COLORS,
        hover_data={
            "title": True,
            "source": True,
            "vader_compound": ":.3f",
            "tb_compound": ":.3f",
        },
        labels={
            "vader_compound": "VADER score",
            "tb_compound":    "TextBlob score",
        },
        opacity=0.7,
        height=440,
    )
    # Perfect agreement diagonal
    fig.add_shape(
        type="line", x0=-1, y0=-1, x1=1, y1=1,
        line=dict(color="gray", dash="dot", width=1),
    )
    # Zero axes
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(150,150,150,0.4)")
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.4)")
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    # ── Source sentiment bar chart ────────────────────────────────────────────
    st.subheader("Average sentiment by source")
    st.caption("Negative = left of zero, positive = right. Both scorers shown side by side.")

    source_avg = (
        filtered.groupby("source")[["vader_compound", "tb_compound"]]
        .mean()
        .round(3)
        .reset_index()
        .sort_values("vader_compound", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=source_avg["source"],
        x=source_avg["vader_compound"],
        name="VADER",
        orientation="h",
        marker_color=[
            LABEL_COLORS[_label(s)] for s in source_avg["vader_compound"]
        ],
    ))
    fig.add_trace(go.Bar(
        y=source_avg["source"],
        x=source_avg["tb_compound"],
        name="TextBlob",
        orientation="h",
        marker_color=TEXTBLOB_COLOR,
        opacity=0.55,
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        barmode="group",
        height=max(280, len(source_avg) * 44),
        margin=dict(t=10, b=10),
        xaxis_title="Compound score",
        xaxis=dict(range=[-1, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width='stretch')

    # ── Top headlines ─────────────────────────────────────────────────────────
    scorer_label = "VADER" if scorer == "vader" else "TextBlob"
    st.subheader(f"Top headlines — {scorer_label}")

    h1, h2 = st.columns(2)

    with h1:
        st.markdown("#### 🟢 Most positive")
        render_headlines(
            get_top_headlines(filtered, "positive", n=n_headlines, scorer=scorer)
        )

    with h2:
        st.markdown("#### 🔴 Most negative")
        render_headlines(
            get_top_headlines(filtered, "negative", n=n_headlines, scorer=scorer)
        )

    # ── Word clouds ───────────────────────────────────────────────────────────
    st.subheader(f"Word clouds — {scorer_label}")
    st.caption(
        "Built from article titles. Word size = frequency within that sentiment group."
    )

    wc1, wc2 = st.columns(2)

    with wc1:
        st.markdown("**Positive titles**")
        st.pyplot(
            make_wordcloud(
                get_wordcloud_text(filtered, "positive", scorer=scorer),
                "positive",
            ),
            width='stretch',
        )

    with wc2:
        st.markdown("**Negative titles**")
        st.pyplot(
            make_wordcloud(
                get_wordcloud_text(filtered, "negative", scorer=scorer),
                "negative",
            ),
            width='stretch',
        )

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("Raw scored data"):
        display_cols = [
            "title", "source", "date",
            "vader_compound", "vader_label",
            "tb_compound", "tb_label", "tb_subjectivity",
        ]
        st.dataframe(
            filtered[display_cols].sort_values("vader_compound"),
            width='stretch',
            hide_index=True,
            column_config={
                "vader_compound": st.column_config.NumberColumn(format="%.3f"),
                "tb_compound":    st.column_config.NumberColumn(format="%.3f"),
                "tb_subjectivity": st.column_config.NumberColumn(format="%.3f"),
                "title":          st.column_config.TextColumn(width="large"),
            },
        )


if __name__ == "__main__":
    main()
