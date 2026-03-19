"""
dashboard.py
------------
Streamlit sentiment dashboard — dark glassmorphism theme.

Sections:
  - KPI strip         : total articles, sources, scorer agreement rate
  - Sentiment donut   : VADER vs TextBlob distribution side by side
  - Scorer scatter    : per-article agreement plot
  - Source bar chart  : average sentiment per source
  - Top headlines     : most positive / negative clickable headlines
  - Word clouds       : title words by sentiment
  - Raw data table    : full scored dataset

Run with:
    uv run streamlit run dashboard.py
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
from database import DB_PATH, init_db, backfill_sources
from sentiment import (
    _label,
    agreement_rate,
    get_top_headlines,
    get_wordcloud_text,
    load_articles,
    score_database,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Newsense · Sentiment Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme constants ───────────────────────────────────────────────────────────

BG          = "#080c14"
GLASS       = "rgba(255,255,255,0.04)"
GLASS_BORDER= "rgba(255,255,255,0.09)"
ACCENT_BLUE = "#4f8ef7"
ACCENT_CYAN = "#00d4ff"
ACCENT_LIME = "#39d98a"
ACCENT_RED  = "#ff4f6b"
ACCENT_AMBER= "#ffb547"
TEXT_PRIMARY= "#e8eaf0"
TEXT_MUTED  = "#7a8099"

LABEL_COLORS = {
    "positive": ACCENT_LIME,
    "neutral":  "#6b7280",
    "negative": ACCENT_RED,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color=TEXT_PRIMARY, size=12),
    margin=dict(t=32, b=16, l=16, r=16),
    legend=dict(
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor=GLASS_BORDER,
        borderwidth=1,
    ),
    hoverlabel=dict(
        bgcolor="#1a2035",
        bordercolor=GLASS_BORDER,
        font=dict(color=TEXT_PRIMARY),
    ),
)


# ── Global CSS injection ──────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: {BG} !important;
        color: {TEXT_PRIMARY};
    }}

    .stApp {{
        background: radial-gradient(ellipse 120% 80% at 20% -10%, rgba(79,142,247,0.12) 0%, transparent 60%),
                    radial-gradient(ellipse 80% 60% at 80% 110%, rgba(0,212,255,0.08) 0%, transparent 55%),
                    {BG};
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: rgba(8,12,20,0.85) !important;
        backdrop-filter: blur(24px);
        border-right: 1px solid {GLASS_BORDER} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {TEXT_PRIMARY};
        font-family: 'Space Grotesk', sans-serif;
    }}

    /* ── Hide default Streamlit chrome but keep sidebar toggle ── */
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    /* Keep the sidebar collapse/expand button fully visible */
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        background: rgba(79,142,247,0.12) !important;
        border: 1px solid rgba(79,142,247,0.3) !important;
        border-radius: 10px !important;
        color: {ACCENT_BLUE} !important;
    }}
    [data-testid="collapsedControl"]:hover {{
        background: rgba(79,142,247,0.22) !important;
    }}

    /* ── Main container padding ── */
    .block-container {{
        padding: 2rem 2.5rem 4rem !important;
        max-width: 1400px;
    }}

    /* ── Page title ── */
    .dash-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {ACCENT_BLUE} 0%, {ACCENT_CYAN} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }}
    .dash-subtitle {{
        color: {TEXT_MUTED};
        font-size: 0.9rem;
        margin-bottom: 1.8rem;
    }}

    /* ── Glass card ── */
    .glass-card {{
        background: {GLASS};
        border: 1px solid {GLASS_BORDER};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: border-color 0.2s ease;
    }}
    .glass-card:hover {{
        border-color: rgba(79,142,247,0.25);
    }}

    /* ── KPI cards ── */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }}
    .kpi-card {{
        background: {GLASS};
        border: 1px solid {GLASS_BORDER};
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(79,142,247,0.3);
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--accent, {ACCENT_BLUE});
        border-radius: 14px 14px 0 0;
        opacity: 0.8;
    }}
    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {TEXT_MUTED};
        margin-bottom: 0.4rem;
    }}
    .kpi-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        line-height: 1;
    }}
    .kpi-sub {{
        font-size: 0.75rem;
        color: {TEXT_MUTED};
        margin-top: 0.3rem;
    }}

    /* ── Section headers ── */
    .section-header {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
        letter-spacing: -0.2px;
        margin: 2rem 0 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .section-caption {{
        font-size: 0.82rem;
        color: {TEXT_MUTED};
        margin-bottom: 1rem;
    }}

    /* ── Divider ── */
    .glass-divider {{
        border: none;
        border-top: 1px solid {GLASS_BORDER};
        margin: 1.6rem 0;
    }}

    /* ── Headline cards ── */
    .headline-card {{
        background: {GLASS};
        border: 1px solid {GLASS_BORDER};
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        transition: background 0.15s, border-color 0.15s;
        backdrop-filter: blur(8px);
    }}
    .headline-card:hover {{
        background: rgba(79,142,247,0.07);
        border-color: rgba(79,142,247,0.25);
    }}
    .headline-title {{
        font-size: 0.88rem;
        font-weight: 500;
        color: {TEXT_PRIMARY};
        text-decoration: none;
        line-height: 1.45;
        display: block;
        margin-bottom: 0.35rem;
    }}
    .headline-title:hover {{ color: {ACCENT_BLUE}; }}
    .headline-meta {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.75rem;
        color: {TEXT_MUTED};
    }}
    .score-pill {{
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'Space Grotesk', monospace;
    }}
    .score-pos {{ background: rgba(57,217,138,0.15); color: {ACCENT_LIME}; }}
    .score-neg {{ background: rgba(255,79,107,0.15); color: {ACCENT_RED};  }}
    .score-neu {{ background: rgba(107,114,128,0.15); color: #9ca3af; }}

    /* ── Streamlit overrides ── */
    div[data-testid="stMetric"] {{
        background: {GLASS} !important;
        border: 1px solid {GLASS_BORDER} !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        backdrop-filter: blur(12px) !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: {TEXT_PRIMARY} !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: {TEXT_MUTED} !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: rgba(79,142,247,0.12) !important;
        color: {ACCENT_BLUE} !important;
        border: 1px solid rgba(79,142,247,0.3) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.18s ease !important;
    }}
    .stButton > button:hover {{
        background: rgba(79,142,247,0.22) !important;
        border-color: {ACCENT_BLUE} !important;
        transform: translateY(-1px) !important;
    }}

    /* ── Selectbox / multiselect / slider ── */
    .stMultiSelect [data-baseweb="tag"] {{
        background: rgba(79,142,247,0.18) !important;
        border: 1px solid rgba(79,142,247,0.3) !important;
        border-radius: 6px !important;
        color: {ACCENT_BLUE} !important;
    }}
    [data-baseweb="select"] > div,
    [data-baseweb="base-input"] > input {{
        background: rgba(255,255,255,0.04) !important;
        border-color: {GLASS_BORDER} !important;
        border-radius: 10px !important;
        color: {TEXT_PRIMARY} !important;
    }}
    .stSlider [data-baseweb="slider"] {{
        padding-top: 0.3rem;
    }}
    .stRadio label {{ color: {TEXT_MUTED} !important; font-size: 0.88rem !important; }}
    .stRadio [aria-checked="true"] + div {{ color: {TEXT_PRIMARY} !important; }}

    /* ── Expander ── */
    [data-testid="stExpander"] {{
        background: {GLASS} !important;
        border: 1px solid {GLASS_BORDER} !important;
        border-radius: 14px !important;
        backdrop-filter: blur(12px) !important;
    }}

    /* ── Dataframe ── */
    .stDataFrame {{ border-radius: 12px; overflow: hidden; }}
    [data-testid="stDataFrameResizable"] {{
        border: 1px solid {GLASS_BORDER} !important;
        border-radius: 12px !important;
    }}

    /* ── Plotly chart container ── */
    [data-testid="stPlotlyChart"] {{
        background: {GLASS};
        border: 1px solid {GLASS_BORDER};
        border-radius: 16px;
        padding: 0.5rem;
        backdrop-filter: blur(12px);
    }}

    /* ── Pyplot / wordcloud container ── */
    [data-testid="stImage"] {{
        background: {GLASS};
        border: 1px solid {GLASS_BORDER};
        border-radius: 16px;
        padding: 0.5rem;
        overflow: hidden;
    }}

    /* ── Toast ── */
    [data-testid="stToast"] {{
        background: rgba(20,28,48,0.95) !important;
        border: 1px solid {GLASS_BORDER} !important;
        border-radius: 12px !important;
        backdrop-filter: blur(16px) !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── Data pipeline ─────────────────────────────────────────────────────────────

def _scrape_fresh() -> bool:
    from database import insert_articles
    from discoverer import discover_all
    from punkt_tab_downloader import ensure_punkt
    from scraper import scrape_articles

    ensure_punkt()
    init_db()
    urls = discover_all(rss_feeds=RSS_FEEDS, sitemap_sites=SITEMAP_SITES,
                        max_per_source=MAX_PER_SOURCE)
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
def _load_cached_data() -> tuple[pd.DataFrame, int]:
    """
    Pure data function — no Streamlit calls allowed inside here.
    Backfills missing sources, scores unscored articles, returns DataFrame.
    """
    backfill_sources()
    newly = score_database()
    return load_articles(), newly


def get_data() -> pd.DataFrame:
    """
    Handles all UI feedback around data loading.
    Scrapes fresh articles if the DB is missing or empty, then loads.
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
            if not _scrape_fresh():
                return pd.DataFrame()
        # Clear cache so the freshly scraped data is picked up
        _load_cached_data.clear()

    df, newly_scored = _load_cached_data()
    if newly_scored:
        st.toast(f"Scored {newly_scored} new articles.", icon="✅")
    return df


# ── Render helpers ────────────────────────────────────────────────────────────

def score_pill_html(score: float) -> str:
    if score > 0.05:
        cls = "score-pos"
        sign = "+"
    elif score < -0.05:
        cls = "score-neg"
        sign = ""
    else:
        cls = "score-neu"
        sign = ""
    return f'<span class="score-pill {cls}">{sign}{score:.3f}</span>'


def render_headlines(headlines: pd.DataFrame) -> None:
    if headlines.empty:
        st.markdown(
            f'<p style="color:{TEXT_MUTED};font-size:0.85rem">Not enough articles.</p>',
            unsafe_allow_html=True,
        )
        return
    for _, row in headlines.iterrows():
        pill = score_pill_html(row["score"])
        st.markdown(f"""
        <div class="headline-card">
            <a href="{row['url']}" target="_blank" class="headline-title">{row['title']}</a>
            <div class="headline-meta">
                <span>{row['source']}</span>
                <span>·</span>
                {pill}
            </div>
        </div>
        """, unsafe_allow_html=True)


def make_wordcloud(text: str, sentiment: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="none")
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)

    if not text.strip():
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                fontsize=12, color=TEXT_MUTED, fontfamily="DM Sans")
        ax.axis("off")
        return fig

    colors = {
        "positive": ["#39d98a", "#22c55e", "#16a34a", "#00d4ff", "#4ade80"],
        "negative": ["#ff4f6b", "#f43f5e", "#e11d48", "#fb7185", "#ff6b8a"],
    }

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "custom", colors.get(sentiment, ["#4f8ef7", "#00d4ff"])
    )

    wc = WordCloud(
        width=900, height=380,
        background_color=None,
        mode="RGBA",
        colormap=cmap,
        max_words=70,
        collocations=False,
        prefer_horizontal=0.8,
        relative_scaling=0.5,
    ).generate(text)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def styled_plotly(fig: go.Figure, height: int = 380) -> go.Figure:
    """Applies the dark glass theme to any Plotly figure."""
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_MUTED),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_MUTED),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="dash-title">🔮 Newsense</div>
        <div class="dash-subtitle">Sentiment intelligence across your scraped news — VADER vs TextBlob</div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1rem;'
            f'font-weight:600;color:{TEXT_PRIMARY};margin-bottom:1rem">Controls</div>',
            unsafe_allow_html=True,
        )

        scorer = st.radio(
            "Active scorer",
            ["vader", "textblob"],
            format_func=lambda x: "VADER" if x == "vader" else "TextBlob",
        )
        n_headlines = st.slider("Headlines per column", 5, 20, 10)

        st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.9rem;'
            f'font-weight:600;color:{TEXT_PRIMARY};margin-bottom:0.8rem">Filters</div>',
            unsafe_allow_html=True,
        )

        min_articles = st.slider(
            "Min articles per source", 1, 20, 1,
            help="Hide sources with fewer articles than this threshold",
        )

        neutral_threshold = st.slider(
            "Neutral boundary (±)",
            min_value=0.00, max_value=0.30, value=0.05, step=0.01,
            help=(
                "Scores between −threshold and +threshold are labelled Neutral. "
                "Raise it to widen the neutral band; lower it to see more positives/negatives."
            ),
        )

        source_placeholder = st.empty()

        st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)
        if st.button("↺  Refresh data", width='stretch',
                     help="Reload articles already in the database"):
            _load_cached_data.clear()
            st.rerun()
        if st.button("🔍  Re-run scraper", width='stretch',
                     help="Discover and scrape fresh articles now"):
            _load_cached_data.clear()
            with st.spinner("Scraping fresh articles..."):
                _scrape_fresh()
            st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading..."):
        df = get_data()

    if df.empty:
        st.warning("No data available. Check your connection and try refreshing.")
        st.stop()

    # Build source list — only show sources with enough articles
    source_counts = df["source"].value_counts()
    sources = sorted([
        s for s in source_counts.index
        if s != "N/A" and source_counts[s] >= min_articles
    ])
    selected = source_placeholder.multiselect("Filter sources", sources, default=sources,
                                               placeholder="All sources",
                                               label_visibility="collapsed")
    filtered = df[df["source"].isin(selected)] if selected else df

    if filtered.empty:
        st.warning("No articles match selected sources.")
        st.stop()

    # Re-apply sentiment labels using the custom neutral threshold
    # This makes all charts, KPIs, and word clouds respond to the slider
    def relabel(score: float) -> str:
        if score > neutral_threshold:
            return "positive"
        elif score < -neutral_threshold:
            return "negative"
        return "neutral"

    filtered = filtered.copy()
    filtered["vader_label"] = filtered["vader_compound"].apply(relabel)
    filtered["tb_label"]    = filtered["tb_compound"].apply(relabel)

    scorer_label = "VADER" if scorer == "vader" else "TextBlob"
    label_col    = "vader_label" if scorer == "vader" else "tb_label"
    score_col    = "vader_compound" if scorer == "vader" else "tb_compound"

    # ── KPI strip ─────────────────────────────────────────────────────────────
    pos_pct   = round((filtered[label_col] == "positive").mean() * 100, 1)
    neg_pct   = round((filtered[label_col] == "negative").mean() * 100, 1)
    agree     = agreement_rate(filtered)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Articles",          f"{len(filtered):,}")
    k2.metric("Sources",           filtered["source"].nunique())
    k3.metric("Positive",          f"{pos_pct}%")
    k4.metric("Negative",          f"{neg_pct}%")
    k5.metric("Scorer agreement",  f"{agree}%",
              help=f"% where VADER and TextBlob assign the same label (threshold ±{neutral_threshold:.2f})")

    st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)

    # ── Sentiment distribution ────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">◈ Sentiment distribution</div>'
        f'<div class="section-caption">How VADER and TextBlob classify the same articles</div>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2)
    for col_widget, col_name, title in [
        (d1, "vader_label", "VADER"),
        (d2, "tb_label",    "TextBlob"),
    ]:
        with col_widget:
            counts = filtered[col_name].value_counts().reset_index()
            counts.columns = ["label", "count"]
            fig = px.pie(
                counts, values="count", names="label",
                color="label", color_discrete_map=LABEL_COLORS,
                hole=0.52,
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont=dict(size=12, color="white"),
                marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)),
            )
            fig = styled_plotly(fig, height=320)
            fig.update_layout(
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{title}</b>",
                    x=0.5, y=0.5, font_size=14,
                    font_color=TEXT_PRIMARY,
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig, width='stretch')

    st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)

    # ── Scatter agreement ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">◈ Scorer agreement</div>'
        f'<div class="section-caption">Each dot is one article — diagonal = both scorers agree · neutral boundary ±{neutral_threshold:.2f}</div>',
        unsafe_allow_html=True,
    )

    fig = px.scatter(
        filtered,
        x="vader_compound",
        y="tb_compound",
        color=label_col,
        color_discrete_map=LABEL_COLORS,
        hover_data={"title": True, "source": True,
                    "vader_compound": ":.3f", "tb_compound": ":.3f"},
        labels={"vader_compound": "VADER score", "tb_compound": "TextBlob score"},
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(0,0,0,0.4)")))
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
                  line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1.5))
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.08)")
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.08)")
    st.plotly_chart(styled_plotly(fig, 420), width='stretch')

    st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)

    # ── Source bar chart ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">◈ Sentiment by source</div>'
        '<div class="section-caption">Average compound score per source — both scorers overlaid</div>',
        unsafe_allow_html=True,
    )

    src = (
        filtered.groupby("source")[["vader_compound", "tb_compound"]]
        .mean().round(3).reset_index()
        .sort_values("vader_compound", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=src["source"], x=src["vader_compound"],
        name="VADER", orientation="h",
        marker=dict(
            color=[LABEL_COLORS[_label(s)] for s in src["vader_compound"]],
            opacity=0.85,
            line=dict(width=0),
        ),
    ))
    fig.add_trace(go.Bar(
        y=src["source"], x=src["tb_compound"],
        name="TextBlob", orientation="h",
        marker=dict(color=ACCENT_BLUE, opacity=0.45, line=dict(width=0)),
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    fig.update_layout(
        barmode="group",
        xaxis=dict(range=[-1, 1]),
        xaxis_title="Compound score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(
        styled_plotly(fig, max(280, len(src) * 46)),
        width='stretch',
    )

    st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)

    # ── Top headlines ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">◈ Top headlines — {scorer_label}</div>'
        f'<div class="section-caption">Most extreme articles scored by {scorer_label}</div>',
        unsafe_allow_html=True,
    )

    h1, h2 = st.columns(2)
    with h1:
        st.markdown(
            f'<div style="color:{ACCENT_LIME};font-size:0.82rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.7rem">'
            f'● Most positive</div>',
            unsafe_allow_html=True,
        )
        render_headlines(
            get_top_headlines(filtered, "positive", n=n_headlines, scorer=scorer)
        )

    with h2:
        st.markdown(
            f'<div style="color:{ACCENT_RED};font-size:0.82rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.7rem">'
            f'● Most negative</div>',
            unsafe_allow_html=True,
        )
        render_headlines(
            get_top_headlines(filtered, "negative", n=n_headlines, scorer=scorer)
        )

    st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)

    # ── Word clouds ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">◈ Word clouds — {scorer_label}</div>'
        '<div class="section-caption">Title words sized by frequency within each sentiment group</div>',
        unsafe_allow_html=True,
    )

    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown(
            f'<div style="color:{ACCENT_LIME};font-size:0.78rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">'
            f'Positive</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(
            make_wordcloud(get_wordcloud_text(filtered, "positive", scorer=scorer), "positive"),
            width='stretch',
        )
    with wc2:
        st.markdown(
            f'<div style="color:{ACCENT_RED};font-size:0.78rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">'
            f'Negative</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(
            make_wordcloud(get_wordcloud_text(filtered, "negative", scorer=scorer), "negative"),
            width='stretch',
        )

    st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("  Raw scored data"):
        st.dataframe(
            filtered[[
                "title", "source", "date",
                "vader_compound", "vader_label",
                "tb_compound", "tb_label", "tb_subjectivity",
            ]].sort_values(score_col),
            width='stretch',
            hide_index=True,
            column_config={
                "vader_compound":  st.column_config.NumberColumn("VADER",     format="%.3f"),
                "tb_compound":     st.column_config.NumberColumn("TextBlob",  format="%.3f"),
                "tb_subjectivity": st.column_config.NumberColumn("Subjectivity", format="%.3f"),
                "vader_label":     st.column_config.TextColumn("V. Label"),
                "tb_label":        st.column_config.TextColumn("TB. Label"),
                "title":           st.column_config.TextColumn("Title", width="large"),
            },
        )


if __name__ == "__main__":
    main()
