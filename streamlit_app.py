# streamlit_app.py
import streamlit as st
import pandas as pd
import glob
import os
import time
from datetime import datetime
import plotly.express as px

# Path must match what your Spark job writes (directory of JSON files)
SENTIMENT_JSON_DIR = "sentiment_output"  # same as your SENTIMENT_JSON_PATH

st.set_page_config(page_title="Real-time News Sentiment Dashboard", layout="wide")

st.title("Real-time News Sentiment Dashboard")
st.markdown("This dashboard reads Spark output JSON files from the `sentiment_output` directory and updates automatically.")

# Auto-refresh interval (seconds)
REFRESH_SECONDS = 5

@st.cache_data(ttl=5)
def load_sentiment_data(json_dir):
    # find all json files written by Spark (part-*.json). If single-file required adjust accordingly.
    if not os.path.exists(json_dir):
        return pd.DataFrame(columns=["text", "polarity", "sentiment", "title", "fetched_at"])

    files = glob.glob(os.path.join(json_dir, "*.json"))
    if len(files) == 0:
        return pd.DataFrame(columns=["text", "polarity", "sentiment", "title", "fetched_at"])

    # read each json file as newline-delimited JSON or normal JSON
    frames = []
    for f in files:
        try:
            # try reading as JSON lines
            df = pd.read_json(f, lines=True)
        except ValueError:
            # fall back to normal json
            try:
                df = pd.read_json(f)
            except Exception:
                continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["text", "polarity", "sentiment", "title", "fetched_at"])
    df = pd.concat(frames, ignore_index=True, sort=False)

    # Normalize column names used in your pipeline (you used 'text' and 'polarity' and 'sentiment')
    # Keep title if present; add fetched_at for ordering (if not present)
    if "title" not in df.columns and "text" in df.columns:
        df["title"] = df["text"].apply(lambda t: (t[:140] + "...") if isinstance(t, str) and len(t) > 140 else t)
    if "fetched_at" not in df.columns:
        df["fetched_at"] = datetime.now().isoformat()

    # ensure columns exist
    for col in ["text", "polarity", "sentiment", "title", "fetched_at"]:
        if col not in df.columns:
            df[col] = None

    # Clean up: keep limited length for display
    df["text_short"] = df["text"].astype(str).str.slice(0, 240)
    return df

# UI: top row: counts and last refresh time
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.subheader("Live refresh")
    st.write(f"Refresh interval: {REFRESH_SECONDS}s")
    if st.button("Refresh now"):
        # clear cache then reload
        load_sentiment_data.clear()
        st.rerun()

with col2:
    st.subheader("Last loaded")
    df_latest = load_sentiment_data(SENTIMENT_JSON_DIR)
    st.write("Records:", len(df_latest))

with col3:
    st.subheader("Status")
    st.write("Directory:", SENTIMENT_JSON_DIR)
    st.write("Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Main panels
df = df_latest.copy()

# Aggregate counts
sentiment_counts = df["sentiment"].fillna("Unknown").value_counts().to_dict()
st.markdown("---")
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Sentiment Distribution (live)")
    if len(df) == 0:
        st.info("No sentiment data found yet. Run the pipeline to generate `sentiment_output` JSON files.")
    else:
        # pie chart
        counts_df = pd.DataFrame(list(sentiment_counts.items()), columns=["sentiment", "count"])
        fig_pie = px.pie(counts_df, names="sentiment", values="count", title="Sentiment Distribution",
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

        # bar chart
        st.subheader("Counts by sentiment")
        fig_bar = px.bar(counts_df.sort_values("count", ascending=False),
                         x="sentiment", y="count", text="count", title="Sentiment Counts")
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Recent Headlines")
    if len(df) == 0:
        st.write("—")
    else:
        # show last 20 entries (assuming fetched_at roughly increases)
        df["fetched_at_parsed"] = pd.to_datetime(df["fetched_at"], errors="coerce")
        df_sorted = df.sort_values(by="fetched_at_parsed", ascending=False).head(20)
        display_df = df_sorted[["title", "text_short", "polarity", "sentiment", "fetched_at"]].rename(
            columns={"title": "Title", "text_short": "Text (short)", "polarity": "Polarity", "sentiment": "Sentiment", "fetched_at": "Fetched At"})
        st.dataframe(display_df, use_container_width=True)

st.markdown("---")
st.subheader("Raw data (first 200 rows)")
if len(df) > 0:
    st.dataframe(df.head(200), use_container_width=True)
else:
    st.write("No data yet.")

# Auto-refresh helper
count = st.rerun if False else None  # placeholder to avoid lint error
st_autorefresh = st.query_params  # just to reference streamlit experimental funcs

# Actual auto-refresh: use st.rerun via timer
# We'll use st.experimental_memo TTL + st.empty loop to wait and rerun
placeholder = st.empty()
with placeholder.container():
    st.write(f"Next refresh in ~{REFRESH_SECONDS} seconds...")
    time.sleep(REFRESH_SECONDS)
    load_sentiment_data.clear()
    st.rerun()