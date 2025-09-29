import streamlit as st
import pandas as pd
import requests, json, os, glob, time
from datetime import datetime
from textblob import TextBlob
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.sql.types import FloatType, StringType

# --------------------------
# CONFIG
# --------------------------
GNEWS_API_KEY = "380af545145de6a25107d08ac9c4ac9c"
QUERY = "stock market"
RAW_JSON_PATH = "news_data.json"
CLEANED_JSON_PATH = "cleaned_news_output"
SENTIMENT_JSON_PATH = "sentiment_output"

# Streamlit page setup
st.set_page_config(page_title="Real-time News Sentiment Dashboard", layout="wide")
st.title("Real-time News Sentiment Dashboard")
st.markdown("This dashboard fetches live news, processes with Spark, and updates automatically.")

# --------------------------
# STEP 1: Fetch news
# --------------------------
def fetch_news():
    url = f"https://gnews.io/api/v4/search?q={QUERY}&lang=en&max=50&token={GNEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article) + "\n")
        return len(articles)
    else:
        st.error(f"Error fetching data: {response.status_code}, {response.text}")
        return 0

# --------------------------
# STEP 2: Preprocessing
# --------------------------
def preprocess_articles(spark):
    df = spark.read.json(RAW_JSON_PATH)
    cleaned_df = df.select("title", "description") \
        .withColumn("text", lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", "")))
    cleaned_df.write.mode("overwrite").json(CLEANED_JSON_PATH)
    return cleaned_df

# --------------------------
# STEP 3: Sentiment Analysis
# --------------------------
def get_sentiment(text):
    if text:
        return TextBlob(text).sentiment.polarity
    return 0.0

def classify_sentiment(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def run_sentiment_analysis(spark):
    sentiment_udf = udf(get_sentiment, FloatType())
    label_udf = udf(classify_sentiment, StringType())
    df = spark.read.json(CLEANED_JSON_PATH)
    df_with_polarity = df.withColumn("polarity", sentiment_udf(df.text))
    scored_df = df_with_polarity.withColumn("sentiment", label_udf(col("polarity"))) \
                                .drop("description")
    scored_df.write.mode("overwrite").json(SENTIMENT_JSON_PATH)
    return scored_df

# --------------------------
# STEP 4: Load Data for Dashboard
# --------------------------
@st.cache_data(ttl=10)
def load_sentiment_data(json_dir):
    if not os.path.exists(json_dir):
        return pd.DataFrame(columns=["text", "polarity", "sentiment", "title"])
    files = glob.glob(os.path.join(json_dir, "*.json"))
    if not files:
        return pd.DataFrame(columns=["text", "polarity", "sentiment", "title"])
    frames = [pd.read_json(f, lines=True) for f in files]
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "title" not in df.columns and "text" in df.columns:
        df["title"] = df["text"].str.slice(0, 100)
    return df

# --------------------------
# Spark session (global)
# --------------------------
spark = SparkSession.builder.appName("RealTimeNewsSentimentPipeline").master("local[*]").getOrCreate()

# --------------------------
# UI Controls
# --------------------------
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.subheader("Controls")
    if st.button("Run Pipeline Now"):
        count = fetch_news()
        if count > 0:
            preprocess_articles(spark)
            run_sentiment_analysis(spark)
        st.cache_data.clear()
        st.rerun()

with col2:
    st.subheader("Last Loaded")
    df_latest = load_sentiment_data(SENTIMENT_JSON_PATH)
    st.write("Records:", len(df_latest))

with col3:
    st.subheader("Status")
    st.write("Directory:", SENTIMENT_JSON_PATH)
    st.write("Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# --------------------------
# Dashboard Display
# --------------------------
df = df_latest.copy()

st.markdown("---")
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Sentiment Distribution (live)")
    if len(df) == 0:
        st.info("No sentiment data yet. Click 'Run Pipeline Now'.")
    else:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        st.bar_chart(counts.set_index("Sentiment"))

with right:
    st.subheader("Recent Headlines")
    if len(df) > 0:
        st.dataframe(df[["title", "polarity", "sentiment"]].tail(20), use_container_width=True)
    else:
        st.write("—")

st.markdown("---")
st.subheader("Raw Data (first 200 rows)")
if len(df) > 0:
    st.dataframe(df.head(200), use_container_width=True)
else:
    st.write("No data yet.")

