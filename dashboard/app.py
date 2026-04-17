"""
Market Sentiment Intelligence Dashboard.
A multi-tab Streamlit dashboard for real-time stock sentiment,
model evaluation, and AI-powered sentiment assistance.
"""

import datetime
import json
import os
import sys
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Streamlit adds the script's directory (dashboard/) to sys.path, not the project root.
# Insert the project root so that `agent`, `aggregation`, etc. are importable.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, ".env"))

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8001")

# Setup Page
st.set_page_config(page_title="Market Sentiment Intelligence", page_icon="", layout="wide")

st.markdown(
    """
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    .stMetric { background-color: rgba(28, 131, 225, 0.05); padding: 12px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; }
    div.stButton > button { border-radius: 6px; }
    .stExpander { border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important; }
</style>
""",
    unsafe_allow_html=True,
)

WINDOW_MAP = {
    "1 hour": ("1h", 1),
    "6 hours": ("6h", 6),
    "24 hours": ("24h", 24),
    "3 days": ("3d", 72),
    "7 days": ("7d", 168),
    "30 days": ("30d", 720),
}

def render_sentiment_chart(symbol: str, window_str: str, hours: int):
    """Render a professional dual-axis sentiment chart."""
    try:
        resp = requests.get(
            f"{API_BASE}/ticker/{symbol}/history",
            params={"hours": hours, "resolution": "auto"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"Could not load chart data: {e}")
        return

    points = data.get("points", [])
    if not points or len(points) < 2:
        st.info("📉 Not enough data points for a chart. Try a longer time window.")
        return

    df = pd.DataFrame(points)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()

    # Bullish/Bearish zones as background shapes
    fig.add_hrect(y0=0.1, y1=1.0, fillcolor="rgba(16,185,129,0.06)", line_width=0,
                  annotation_text="Bullish Zone", annotation_position="top left",
                  annotation_font_color="rgba(16,185,129,0.5)")
    fig.add_hrect(y0=-1.0, y1=-0.1, fillcolor="rgba(239,68,68,0.06)", line_width=0,
                  annotation_text="Bearish Zone", annotation_position="bottom left",
                  annotation_font_color="rgba(239,68,68,0.5)")

    # Neutral reference line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.4)", line_width=1)

    # Article count bars (secondary axis)
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["doc_count"],
        name="Articles",
        yaxis="y2",
        marker_color="rgba(99,102,241,0.25)",
        hovertemplate="Articles: %{y}<extra></extra>",
    ))

    # Sentiment score line (primary)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["score"],
        mode="lines+markers",
        name="Sentiment",
        line=dict(color="#10B981", width=2.5),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.08)",
        hovertemplate="Score: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Sentiment Trend — {symbol}", font=dict(size=16)),
        yaxis=dict(
            title="Sentiment Score",
            range=[-1, 1],
            tickvals=[-1, -0.5, -0.1, 0, 0.1, 0.5, 1],
            gridcolor="rgba(150,150,150,0.15)",
            zerolinecolor="rgba(150,150,150,0.3)",
        ),
        yaxis2=dict(
            title="Article Count",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis=dict(gridcolor="rgba(150,150,150,0.15)"),
        hovermode="x unified",
        height=420,
        margin=dict(l=50, r=50, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12),
    )

    st.plotly_chart(fig, use_container_width=True)

def render_attribution_visualization(attrs: list[dict], headline: str, score: float = None):
    """
    Render a professional token attribution visualization.
    Shows: color-coded headline, attribution bar chart, and interpretation summary.
    """
    if not attrs:
        st.write(headline)
        return

    # Normalize attributions to [-1, 1]
    max_abs = max(abs(a["attribution"]) for a in attrs) or 1.0
    normalized = [(a["token"], a["attribution"] / max_abs) for a in attrs]

    # Part 1: Color-coded headline
    html_parts = []
    for tok, norm_attr in normalized:
        abs_attr = abs(norm_attr)
        if abs_attr > 0.15:
            opacity = min(abs_attr * 0.8, 0.85)
            if score is not None and score > 0.05:
                bg = f"rgba(16,185,129,{opacity})"
            elif score is not None and score < -0.05:
                bg = f"rgba(239,68,68,{opacity})"
            else:
                bg = f"rgba(99,102,241,{opacity})"
            text_color = "white" if opacity > 0.5 else "inherit"
        else:
            bg = "rgba(150,150,150,0.1)"
            text_color = "inherit"
        html_parts.append(
            f'<span style="background:{bg};color:{text_color};padding:4px 8px;'
            f'border-radius:6px;margin:2px;display:inline-block;'
            f'font-size:15px;font-weight:500;line-height:2.2">{tok}</span>'
        )

    st.markdown(
        f'<div style="margin:12px 0;padding:16px;background:rgba(255,255,255,0.03);'
        f'border-radius:10px;border:1px solid rgba(255,255,255,0.08)">'
        f'{"".join(html_parts)}</div>',
        unsafe_allow_html=True,
    )

    # Part 2: Horizontal bar chart — top 10 by absolute attribution
    _punct = {",", ".", ":", ";", "-", "'", '"', "(", ")", "[", "]", "!", "?", "##"}
    meaningful = [(t, v) for t, v in normalized if len(t) > 1 and t not in _punct]

    sorted_attrs = sorted(meaningful, key=lambda x: abs(x[1]), reverse=True)[:10]
    sorted_attrs.reverse()  # bottom = highest for horizontal chart readability

    tokens = [t for t, _ in sorted_attrs]
    values = [v for _, v in sorted_attrs]
    if score is not None and score > 0.05:
        colors = ["#10B981" if abs(v) > 0.1 else "#6B7280" for v in values]
    elif score is not None and score < -0.05:
        colors = ["#EF4444" if abs(v) > 0.1 else "#6B7280" for v in values]
    else:
        colors = ["#6B7280" for v in values]

    fig = go.Figure(go.Bar(
        x=[abs(v) for v in values],
        y=tokens,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Token Attribution Scores (Top 10)", font=dict(size=14)),
        xaxis=dict(
            title="Attribution strength (longer bar = stronger influence on prediction)",
            range=[0, 1.1],
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.4)",
            gridcolor="rgba(150,150,150,0.1)",
        ),
        yaxis=dict(tickfont=dict(size=12)),
        height=max(250, len(sorted_attrs) * 30 + 100),
        margin=dict(l=120, r=20, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Part 3: Auto-generated interpretation
    top_drivers = sorted(meaningful, key=lambda x: abs(x[1]), reverse=True)[:5]
    top_positive = [(t, v) for t, v in top_drivers] if score is not None and score > 0.05 else []
    top_negative = [(t, v) for t, v in top_drivers] if score is not None and score < -0.05 else []

    interpretation_parts = []
    if top_positive:
        pos_words = ", ".join(
            f"**{t}**" for t, _ in sorted(top_positive, key=lambda x: -x[1])[:3]
        )
        interpretation_parts.append(f"Bullish signals: {pos_words}")
    if top_negative:
        neg_words = ", ".join(
            f"**{t}**" for t, _ in sorted(top_negative, key=lambda x: x[1])[:3]
        )
        interpretation_parts.append(f"Bearish signals: {neg_words}")

    if score is not None:
        if score > 0.1:
            verdict = "📈 **Overall: Bullish**"
        elif score < -0.1:
            verdict = "📉 **Overall: Bearish**"
        else:
            verdict = "➡️ **Overall: Neutral**"
        interpretation_parts.append(verdict)

    if interpretation_parts:
        st.markdown(" | ".join(interpretation_parts))

    st.caption(
        "Attribution computed using Integrated Gradients (Sundararajan et al., 2017) "
        "over the FinBERT transformer backbone."
    )


tab_dashboard, tab_evaluation, tab_chat = st.tabs([
    "Sentiment Dashboard", 
    "Model Evaluation", 
    "AI Assistant"
])

with tab_dashboard:
    st.header("Market Sentiment Intelligence")
    st.subheader("Real-time NLP-powered sentiment analysis for NSE/BSE stocks")

    col_search, col_window = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("Search for a ticker (e.g. Reliance, Idea, Tata)", value="")
    with col_window:
        selected_window_label = st.selectbox("Time Window", list(WINDOW_MAP.keys()), index=4)

    window_str, window_hours = WINDOW_MAP[selected_window_label]

    if search_query:
        # call search endpoint
        try:
            r = requests.get(f"{API_BASE}/search", params={"q": search_query})
            r.raise_for_status()
            results = r.json().get("results", [])
            options = {res["symbol"]: f"{res['symbol']} - {res.get('name', res['symbol'])}" for res in results}
        except Exception:
            options = {}
            st.error("Error connecting to search API.")
        
        if options:
            selected_ticker = st.selectbox("Select Ticker", list(options.keys()), format_func=lambda x: options[x])
            
            if selected_ticker:
                st.markdown("---")
                try:
                    with st.spinner("Fetching sentiment data..."):
                        r2 = requests.get(f"{API_BASE}/ticker/{selected_ticker}", params={"window": window_str})
                        r2.raise_for_status()
                        ticker_data = r2.json()
                except Exception as e:
                    st.error(f"Error fetching ticker data: {e}")
                    ticker_data = {}

                if ticker_data:
                    if ticker_data.get("status") == "insufficient_data":
                        st.warning("Insufficient data available for this ticker in the selected time window.")
                    else:
                        if ticker_data.get("live_fetched"):
                            st.caption("⚡ Fresh data fetched in real-time")
                            
                        # Metrics Layout
                        m1, m2, m3 = st.columns(3)
                        
                        score = ticker_data.get("score", 0)
                        if score > 0.1:
                            delta = "Bullish"
                            delta_color = "normal"
                        elif score < -0.1:
                            delta = "Bearish"
                            delta_color = "inverse"
                        else:
                            delta = "Neutral"
                            delta_color = "off"
                            
                        with m1:
                            st.metric("Sentiment Score", f"{score:+.3f}", delta=delta, delta_color=delta_color)
                        with m2:
                            st.metric("Article Volume", ticker_data.get("doc_count", 0), 
                                     delta=f"Confidence: {ticker_data.get('confidence', 'Unknown').capitalize()}", 
                                     delta_color="off")
                        with m3:
                            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                            st.metric("Time Window", selected_window_label, delta=f"As of {now_str}", delta_color="off")
                            
                        st.markdown("---")
                        render_sentiment_chart(selected_ticker, window_str, window_hours)
                        
                        st.markdown("---")
                        st.subheader("Recent Articles")
                        articles = ticker_data.get("articles", [])
                        
                        col_arts, col_sources = st.columns([2, 1])
                        
                        with col_arts:
                            for idx, article in enumerate(articles):
                                hd = article.get("headline", "")
                                source = article.get("source_id", "")
                                art_score = article.get("sentiment", {}).get("score", 0)
                                pub = article.get("published_at", "")[:10]
                                
                                color = "🟢" if art_score > 0.1 else "🔴" if art_score < -0.1 else "⚪"
                                with st.expander(f"{color} {hd}  ({source})"):
                                    st.write(f"**Date:** {pub} | **Score:** {art_score:+.3f}")
                                    st.markdown(f"[Read Article]({article.get('url', '#')})")
                                    
                                    if st.button("Show word attribution", key=f"attr_{idx}_{article.get('id')}"):
                                        with st.spinner("Analyzing..."):
                                            try:
                                                exp_r = requests.get(f"{API_BASE}/explain/{article.get('id')}")
                                                if exp_r.status_code == 200:
                                                    attrs = exp_r.json().get("attributions", [])
                                                    st.markdown("#### Explainability")
                                                    render_attribution_visualization(attrs, hd, art_score)
                                                else:
                                                    st.error("Explainability not available for this article.")
                                            except Exception:
                                                st.error("Error contacting explainer API.")
                                                
                        with col_sources:
                            st.subheader("Source Breakdown")
                            source_counts = {}
                            for a in articles:
                                src = a.get("source_id", "Unknown")
                                source_counts[src] = source_counts.get(src, 0) + 1
                                
                            if source_counts:
                                src_df = pd.DataFrame(list(source_counts.items()), columns=["Source", "Count"])
                                src_fig = px.bar(src_df, x="Count", y="Source", orientation="h")
                                st.plotly_chart(src_fig, use_container_width=True)


with tab_evaluation:
    st.header("Model Evaluation")
    st.markdown("Deep learning model metrics and explainability tools for FinBERT.")
    
    st.subheader("Model Comparison")
    mc_path = "evaluation/model_comparison.json"
    if os.path.exists(mc_path):
        try:
            with open(mc_path, "r") as f:
                mc_data = json.load(f)
            mc_df = pd.DataFrame(mc_data).T.reset_index().rename(columns={"index": "Model"})
            st.dataframe(mc_df, hide_index=True, use_container_width=True)
            if os.path.exists("evaluation/plots/model_comparison.png"):
                st.image("evaluation/plots/model_comparison.png")
        except Exception as e:
            st.error(f"Error loading model comparison: {e}")
    else:
        st.warning("Model comparison data not yet generated.")

    col_cm, col_tc = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix")
        if os.path.exists("evaluation/plots/confusion_matrix.png"):
            st.image("evaluation/plots/confusion_matrix.png", use_container_width=True)
        else:
            st.warning("Confusion matrix not yet generated.")
            
    with col_tc:
        st.subheader("Training Curves")
        if os.path.exists("evaluation/plots/training_curves.png"):
            st.image("evaluation/plots/training_curves.png", use_container_width=True)
        else:
            st.warning("Training curves not yet generated.")

    st.markdown("---")
    st.subheader("Dataset Statistics")
    
    gold_count = 0
    silver_count = 0
    cat_counts = {}
    try:
        if os.path.exists("data/labeled_headlines.csv"):
            df_g = pd.read_csv("data/labeled_headlines.csv")
            gold_count = len(df_g)
            for c, v in df_g["label"].value_counts().items():
                cat_counts[c] = cat_counts.get(c, 0) + v
                
        if os.path.exists("data/silver_labels.csv"):
            df_s = pd.read_csv("data/silver_labels.csv")
            silver_count = len(df_s)
            for c, v in df_s["label"].value_counts().items():
                cat_counts[c] = cat_counts.get(c, 0) + v
        
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            st.metric("Total Labeled Documents", gold_count + silver_count)
        with sm2:
            st.metric("Gold Labels (Manual)", gold_count)
        with sm3:
            st.metric("Silver Labels (Auto)", silver_count)
            
        st.write("Class distribution:")
        st.write(cat_counts)
    except Exception as e:
        st.error(f"Error loading dataset stats: {e}")

    st.markdown("---")
    st.subheader("Live Explainability Demo")
    st.write("Understand how the FinBERT model attributes sentiment to specific words.")
    
    demo_text = st.text_input("Enter a financial headline:", "Reliance Industries reports record quarterly profit amid strong demand")
    if st.button("Explain"):
        try:
            from nlp.explainer import explain_text
            with st.spinner("Calculating word attributions..."):
                ex_attrs = explain_text(demo_text, n_steps=30)  # Returns list[dict]

                render_attribution_visualization(ex_attrs, demo_text)
        except Exception as e:
            st.error(f"Error running explainer: {e}")

with tab_chat:
    st.header("Sentiment AI Assistant")
    st.caption("Ask questions about stock market sentiment. Powered by Gemini + live sentiment data.")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about any stock sentiment... (e.g., 'How is Reliance doing?')"):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sentiment data..."):
                try:
                    from agent.chat import chat as gemini_chat
                    response = gemini_chat(prompt, st.session_state.chat_messages[:-1])
                except ImportError:
                    response = "⚠️ Chat module not found. Ensure agent/chat.py exists."
                except Exception as e:
                    response = f"⚠️ Error: {str(e)}"
                st.markdown(response)
        
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
