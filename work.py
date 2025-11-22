import requests
import json
import os
import time
from datetime import datetime
from textblob import TextBlob
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf, lit, current_timestamp
from pyspark.sql.types import FloatType, StringType, StructType, StructField
import matplotlib.pyplot as plt

# --------------------------
# CONFIG
# --------------------------
GNEWS_API_KEY = '380af545145de6a25107d08ac9c4ac9c'
QUERY = 'stock market'
RAW_JSON_PATH = "news_data.json"
CLEANED_JSON_PATH = "cleaned_news_output"
SENTIMENT_JSON_PATH = "sentiment_output"
FETCH_INTERVAL = 300  # Fetch news every 5 minutes

# --------------------------
# STEP 1: Fetch news
# --------------------------
def fetch_news(append_mode=False):
    """
    Fetch news from GNews API
    If append_mode=True, appends to existing file
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching News from GNews API...")
    url = f"https://gnews.io/api/v4/search?q={QUERY}&lang=en&max=50&token={GNEWS_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            
            # Add timestamp to each article
            for article in articles:
                article['fetched_at'] = datetime.now().isoformat()
            
            mode = "a" if append_mode and os.path.exists(RAW_JSON_PATH) else "w"
            with open(RAW_JSON_PATH, mode, encoding="utf-8") as f:
                for article in articles:
                    f.write(json.dumps(article) + "\n")
            
            print(f"✅ Fetched and saved {len(articles)} articles.")
            return len(articles)
        else:
            print(f"❌ Error fetching data: {response.status_code}, {response.text}")
            return 0
    except Exception as e:
        print(f"❌ Exception during fetch: {e}")
        return 0

# --------------------------
# STEP 2: Preprocessing
# --------------------------
def preprocess_articles(spark):
    """
    Clean and preprocess news articles using PySpark
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preprocessing News Data with PySpark...")
    
    if not os.path.exists(RAW_JSON_PATH):
        print("⚠️ No raw data found. Run fetch_news() first.")
        return None
    
    try:
        df = spark.read.json(RAW_JSON_PATH)
        
        # Combine title and description for better sentiment analysis
        cleaned_df = df.select(
            "title", 
            "description",
            "url",
            "publishedAt",
            "fetched_at"
        ).withColumn(
            "text", 
            lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", ""))
        ).withColumn(
            "title_clean",
            lower(regexp_replace(col("title"), "[^a-zA-Z\\s]", ""))
        )
        
        # Filter out null or empty descriptions
        cleaned_df = cleaned_df.filter(col("text").isNotNull() & (col("text") != ""))
        
        cleaned_df.write.mode("overwrite").json(CLEANED_JSON_PATH)
        print(f"✅ Saved cleaned data to {CLEANED_JSON_PATH}")
        return cleaned_df
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return None

# --------------------------
# STEP 3: Sentiment Analysis
# --------------------------
def get_sentiment(text):
    """Calculate sentiment polarity using TextBlob"""
    if text and isinstance(text, str) and len(text.strip()) > 0:
        try:
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0
    return 0.0

def classify_sentiment(score):
    """Classify sentiment based on polarity score"""
    if score is None:
        return "Neutral"
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def run_sentiment_analysis(spark):
    """
    Perform sentiment analysis on preprocessed data
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running Sentiment Analysis...")
    
    if not os.path.exists(CLEANED_JSON_PATH):
        print("⚠️ No cleaned data found. Run preprocess_articles() first.")
        return None
    
    try:
        sentiment_udf = udf(get_sentiment, FloatType())
        label_udf = udf(classify_sentiment, StringType())
        
        df = spark.read.json(CLEANED_JSON_PATH)
        
        # Calculate sentiment on combined text (title + description)
        df_with_polarity = df.withColumn(
            "combined_text", 
            col("title_clean") + " " + col("text")
        ).withColumn(
            "polarity", 
            sentiment_udf(col("combined_text"))
        )
        
        scored_df = df_with_polarity.withColumn(
            "sentiment", 
            label_udf(col("polarity"))
        ).select(
            "title",
            "description",
            "url",
            "text",
            "polarity",
            "sentiment",
            "publishedAt",
            "fetched_at"
        )
        
        # Write to output directory
        scored_df.coalesce(1).write.mode("overwrite").json(SENTIMENT_JSON_PATH)
        
        # Also write a single consolidated JSON file for easier dashboard reading
        pandas_df = scored_df.toPandas()
        pandas_df.to_json(
            os.path.join(SENTIMENT_JSON_PATH, "consolidated.json"),
            orient="records",
            lines=True,
            force_ascii=False
        )
        
        print(f"✅ Saved sentiment results to '{SENTIMENT_JSON_PATH}'")
        print(f"\n📊 Sample Results:")
        scored_df.select("title", "polarity", "sentiment").show(10, truncate=60)
        
        return scored_df
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        return None

# --------------------------
# STEP 4: Summary Statistics
# --------------------------
def print_sentiment_summary(scored_df):
    """Print sentiment distribution summary"""
    if scored_df is None:
        print("⚠️ No data available for summary.")
        return
    
    print(f"\n📊 Sentiment Summary [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:")
    print("=" * 50)
    
    counts = scored_df.groupBy("sentiment").count().collect()
    total = sum(row['count'] for row in counts)
    
    for row in counts:
        percentage = (row['count'] / total * 100) if total > 0 else 0
        print(f"{row['sentiment']:>10}: {row['count']:>4} articles ({percentage:>5.1f}%)")
    print("=" * 50)

# --------------------------
# STEP 5: Visualization
# --------------------------
def plot_sentiment_pie(scored_df):
    """Generate pie chart of sentiment distribution"""
    if scored_df is None:
        print("⚠️ No data available for plotting.")
        return
    
    print(f"\n📈 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating Pie Chart Visualization...")
    
    try:
        counts = scored_df.groupBy("sentiment").count().collect()
        sentiment_counts = {row["sentiment"]: row["count"] for row in counts}
        
        labels = []
        sizes = []
        colors_map = {"Positive": "#66BC6A", "Neutral": "#FFA726", "Negative": "#EF5350"}
        colors = []
        
        for sentiment in ["Positive", "Neutral", "Negative"]:
            count = sentiment_counts.get(sentiment, 0)
            if count > 0:
                labels.append(f"{sentiment}")
                sizes.append(count)
                colors.append(colors_map[sentiment])
        
        if sizes:
            plt.figure(figsize=(8, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
            plt.title(f"News Sentiment Distribution\n({sum(sizes)} articles)")
            plt.axis("equal")
            plt.tight_layout()
            
            # Save plot
            plt.savefig("sentiment_distribution.png", dpi=150, bbox_inches='tight')
            print("✅ Chart saved as 'sentiment_distribution.png'")
            plt.show()
        else:
            print("⚠️ No data to plot.")
    except Exception as e:
        print(f"❌ Error creating plot: {e}")

# --------------------------
# CONTINUOUS STREAMING MODE
# --------------------------
def run_continuous_pipeline(spark, iterations=None):
    """
    Run the pipeline continuously, fetching new data periodically
    iterations: Number of iterations to run (None = infinite)
    """
    print("\n" + "="*60)
    print("🚀 Starting Continuous News Sentiment Pipeline")
    print("="*60)
    
    iteration = 0
    while True:
        iteration += 1
        if iterations and iteration > iterations:
            print(f"\n✅ Completed {iterations} iterations. Stopping.")
            break
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")
        
        # Fetch new articles
        articles_count = fetch_news(append_mode=False)
        
        if articles_count > 0:
            # Process pipeline
            cleaned_df = preprocess_articles(spark)
            if cleaned_df is not None:
                scored_df = run_sentiment_analysis(spark)
                if scored_df is not None:
                    print_sentiment_summary(scored_df)
        
        if iterations is None or iteration < iterations:
            print(f"\n⏳ Waiting {FETCH_INTERVAL} seconds before next fetch...")
            print(f"   (Next run at: {datetime.fromtimestamp(time.time() + FETCH_INTERVAL).strftime('%Y-%m-%d %H:%M:%S')})")
            time.sleep(FETCH_INTERVAL)

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    import sys
    
    spark = SparkSession.builder \
        .appName("RealTimeNewsSentimentPipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # Check command line arguments
        mode = sys.argv[1] if len(sys.argv) > 1 else "once"
        
        if mode == "continuous":
            # Run continuously
            iterations = int(sys.argv[2]) if len(sys.argv) > 2 else None
            run_continuous_pipeline(spark, iterations=iterations)
        else:
            # Run once
            print("\n🔹 Running pipeline once...")
            fetch_news()
            preprocess_articles(spark)
            scored_df = run_sentiment_analysis(spark)
            print_sentiment_summary(scored_df)
            plot_sentiment_pie(scored_df)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
    finally:
        spark.stop()
        print("\n✅ Spark session closed.")