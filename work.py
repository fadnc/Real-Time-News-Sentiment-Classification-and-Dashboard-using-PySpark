import requests
import json
import os
from textblob import TextBlob
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.sql.types import FloatType, StringType
import matplotlib.pyplot as plt

# --------------------------
# CONFIG
# --------------------------
GNEWS_API_KEY = '380af545145de6a25107d08ac9c4ac9c'
QUERY = 'stock market'
RAW_JSON_PATH = "news_data.json"
CLEANED_JSON_PATH = "cleaned_news_output"
SENTIMENT_JSON_PATH = "sentiment_output"


# --------------------------
# STEP 1: Fetch news
# --------------------------
def fetch_news():
    print("\n[Step 1] Fetching News from GNews API...")
    url = f"https://gnews.io/api/v4/search?q={QUERY}&lang=en&max=50&token={GNEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])

        with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article) + "\n")

        print(f"✅ Fetched and saved {len(articles)} articles.")
    else:
        print(f"❌ Error fetching data: {response.status_code}, {response.text}")


# --------------------------
# STEP 2: Preprocessing
# --------------------------
def preprocess_articles(spark):
    print("\n[Step 2] Preprocessing News Data with PySpark...")
    df = spark.read.json(RAW_JSON_PATH)
    cleaned_df = df.select("title", "description") \
        .withColumn("text", lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", "")))

    cleaned_df.write.mode("overwrite").json(CLEANED_JSON_PATH)
    print(f"✅ Saved cleaned data to {CLEANED_JSON_PATH}")
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
    print("\n[Step 3] Running Sentiment Analysis...")

    sentiment_udf = udf(get_sentiment, FloatType())
    label_udf = udf(classify_sentiment, StringType())

    df = spark.read.json(CLEANED_JSON_PATH)

    df_with_polarity = df.withColumn("polarity", sentiment_udf(df.text))
    scored_df = df_with_polarity.withColumn("sentiment", label_udf(col("polarity"))) \
                                .drop("description")

    scored_df.write.mode("overwrite").json(SENTIMENT_JSON_PATH)
    print(f"✅ Saved sentiment results to '{SENTIMENT_JSON_PATH}'")
    scored_df.select("text", "polarity", "sentiment").show(truncate=100)

    return scored_df


# --------------------------
# STEP 4: Simple Summary
# --------------------------
def print_sentiment_summary(scored_df):
    counts = scored_df.groupBy("sentiment").count().collect()
    print("\n📊 Sentiment Summary:")
    for row in counts:
        print(f"{row['sentiment']}: {row['count']} articles")


# --------------------------
# STEP 5: Visualization
# --------------------------
def plot_sentiment_pie(scored_df):
    print("\n📈 [Step 5] Generating Pie Chart Visualization...")

    counts = scored_df.groupBy("sentiment").count().collect()
    sentiment_counts = {row["sentiment"]: row["count"] for row in counts}

    labels = []
    sizes = []
    colors = {"Positive": "#66BC6A", "Neutral": "#FFA726", "Negative": "#EF5350"}

    for sentiment in ["Positive", "Neutral", "Negative"]:
        count = sentiment_counts.get(sentiment, 0)
        if count > 0:
            labels.append(f"{sentiment} ({count})")
            sizes.append(count)

    if sizes:
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                startangle=140, colors=[colors[label.split()[0]] for label in labels])
        plt.title("News Sentiment Distribution")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No data to plot.")


# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("RealTimeNewsSentimentPipeline") \
        .master("local[*]") \
        .getOrCreate()

    fetch_news()
    preprocess_articles(spark)
    scored_df = run_sentiment_analysis(spark)
    print_sentiment_summary(scored_df)
    plot_sentiment_pie(scored_df)

    spark.stop()
