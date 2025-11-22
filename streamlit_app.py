import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import time
from datetime import datetime, timedelta

# Path must match what your Spark job writes
SENTIMENT_JSON_DIR = "sentiment_output"

st.set_page_config(
    page_title="Real-time News Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .positive { color: #66BC6A; }
    .negative { color: #EF5350; }
    .neutral { color: #FFA726; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Real-time News Sentiment Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Dashboard Settings")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=60,
        value=10,
        step=5
    )
    
    st.markdown("---")
    
    # Manual refresh button
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    # Display options
    st.subheader("Display Options")
    show_raw_data = st.checkbox("Show Raw Data", value=False)
    num_recent = st.slider("Recent headlines to show", 5, 50, 20)
    
    st.markdown("---")
    st.info("💡 **Tip**: The dashboard updates automatically when new data is available.")

# Function to load sentiment data
@st.cache_data(ttl=10)
def load_sentiment_data(json_dir):
    """Load sentiment analysis results from JSON files"""
    if not os.path.exists(json_dir):
        return pd.DataFrame()
    
    # Try to load consolidated file first
    consolidated_file = os.path.join(json_dir, "consolidated.json")
    if os.path.exists(consolidated_file):
        try:
            df = pd.read_json(consolidated_file, lines=True)
            if not df.empty:
                return process_dataframe(df)
        except Exception as e:
            st.warning(f"Error reading consolidated file: {e}")
    
    # Fall back to reading all JSON files
    files = glob.glob(os.path.join(json_dir, "*.json"))
    if not files:
        return pd.DataFrame()
    
    frames = []
    for f in files:
        if "consolidated.json" in f:
            continue
        try:
            df = pd.read_json(f, lines=True)
            frames.append(df)
        except:
            try:
                df = pd.read_json(f)
                frames.append(df)
            except:
                continue
    
    if not frames:
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True, sort=False)
    return process_dataframe(df)

def process_dataframe(df):
    """Process and clean the dataframe"""
    # Ensure required columns exist
    required_cols = ["title", "text", "polarity", "sentiment"]
    for col in required_cols:
        if col not in df.columns:
            if col == "text" and "description" in df.columns:
                df["text"] = df["description"]
            else:
                df[col] = None
    
    # Add display columns
    if "text" in df.columns:
        df["text_short"] = df["text"].astype(str).str.slice(0, 200) + "..."
    else:
        df["text_short"] = ""
    
    # Parse timestamps
    if "fetched_at" in df.columns:
        df["fetched_at_parsed"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    else:
        df["fetched_at_parsed"] = pd.Timestamp.now()
    
    if "publishedAt" in df.columns:
        df["publishedAt_parsed"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    
    # Fill missing sentiments
    df["sentiment"] = df["sentiment"].fillna("Neutral")
    
    # Remove duplicates based on title
    df = df.drop_duplicates(subset=["title"], keep="first")
    
    return df

# Load data
df = load_sentiment_data(SENTIMENT_JSON_DIR)

# Display update time
col_time1, col_time2, col_time3 = st.columns([1, 1, 1])
with col_time1:
    st.metric("📅 Last Updated", datetime.now().strftime("%H:%M:%S"))
with col_time2:
    st.metric("📰 Total Articles", len(df))
with col_time3:
    if not df.empty and "fetched_at_parsed" in df.columns:
        latest = df["fetched_at_parsed"].max()
        if pd.notna(latest):
            st.metric("🕐 Latest Article", latest.strftime("%H:%M:%S"))

st.markdown("---")

# Main content
if df.empty:
    st.warning("⚠️ No sentiment data found yet.")
    st.info("""
    **To generate data:**
    1. Run `python work.py` to fetch and analyze news
    2. Or run `python work.py continuous` for continuous updates
    3. The dashboard will automatically display results
    """)
else:
    # Calculate sentiment statistics
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    total = len(df)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        positive_count = sentiment_counts.get("Positive", 0)
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        st.metric(
            "✅ Positive",
            f"{positive_count}",
            f"{positive_pct:.1f}%",
            delta_color="normal"
        )
    
    with col2:
        neutral_count = sentiment_counts.get("Neutral", 0)
        neutral_pct = (neutral_count / total * 100) if total > 0 else 0
        st.metric(
            "➖ Neutral",
            f"{neutral_count}",
            f"{neutral_pct:.1f}%"
        )
    
    with col3:
        negative_count = sentiment_counts.get("Negative", 0)
        negative_pct = (negative_count / total * 100) if total > 0 else 0
        st.metric(
            "❌ Negative",
            f"{negative_count}",
            f"{negative_pct:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        avg_polarity = df["polarity"].mean() if "polarity" in df.columns else 0
        st.metric(
            "📊 Avg Polarity",
            f"{avg_polarity:.3f}",
            "Positive" if avg_polarity > 0 else "Negative"
        )
    
    st.markdown("---")
    
    # Charts row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📊 Sentiment Distribution")
        
        # Prepare data for pie chart
        counts_df = pd.DataFrame([
            {"Sentiment": k, "Count": v} 
            for k, v in sentiment_counts.items()
        ])
        
        # Color mapping
        color_map = {
            "Positive": "#66BC6A",
            "Neutral": "#FFA726",
            "Negative": "#EF5350"
        }
        
        fig_pie = px.pie(
            counts_df,
            names="Sentiment",
            values="Count",
            title="",
            hole=0.4,
            color="Sentiment",
            color_discrete_map=color_map
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col2:
        st.subheader("📈 Sentiment Counts")
        
        # Bar chart
        counts_df_sorted = counts_df.sort_values("Count", ascending=False)
        
        fig_bar = px.bar(
            counts_df_sorted,
            x="Sentiment",
            y="Count",
            text="Count",
            color="Sentiment",
            color_discrete_map=color_map
        )
        fig_bar.update_traces(
            textposition="outside",
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
        fig_bar.update_layout(
            showlegend=False,
            height=400,
            yaxis_title="Number of Articles",
            xaxis_title=""
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Polarity distribution
    if "polarity" in df.columns and df["polarity"].notna().any():
        st.subheader("📉 Polarity Score Distribution")
        
        fig_hist = px.histogram(
            df,
            x="polarity",
            nbins=30,
            title="",
            labels={"polarity": "Polarity Score", "count": "Frequency"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig_hist.update_layout(
            showlegend=False,
            height=300,
            bargap=0.1
        )
        fig_hist.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Neutral"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Recent headlines
    st.subheader(f"📰 Recent Headlines (Last {num_recent})")
    
    # Sort by timestamp and get recent articles
    df_sorted = df.sort_values(by="fetched_at_parsed", ascending=False).head(num_recent)
    
    # Display as cards
    for idx, row in df_sorted.iterrows():
        sentiment = row.get("sentiment", "Neutral")
        polarity = row.get("polarity", 0)
        title = row.get("title", "No title")
        text_short = row.get("text_short", "")
        url = row.get("url", "")
        
        # Sentiment emoji and color
        if sentiment == "Positive":
            emoji = "✅"
            color_class = "positive"
        elif sentiment == "Negative":
            emoji = "❌"
            color_class = "negative"
        else:
            emoji = "➖"
            color_class = "neutral"
        
        with st.container():
            col_a, col_b = st.columns([5, 1])
            
            with col_a:
                st.markdown(f"**{emoji} {title}**")
                if text_short:
                    st.caption(text_short)
                if url:
                    st.markdown(f"[Read more]({url})")
            
            with col_b:
                st.markdown(f'<div class="{color_class}" style="font-size: 1.2rem; font-weight: bold; text-align: center;">{sentiment}</div>', unsafe_allow_html=True)
                st.caption(f"Score: {polarity:.3f}")
            
            st.markdown("---")
    
    # Raw data table (optional)
    if show_raw_data:
        st.subheader("🗂️ Raw Data")
        display_cols = ["title", "sentiment", "polarity", "text_short", "fetched_at"]
        display_df = df[[col for col in display_cols if col in df.columns]]
        st.dataframe(display_df, use_container_width=True, height=400)

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()