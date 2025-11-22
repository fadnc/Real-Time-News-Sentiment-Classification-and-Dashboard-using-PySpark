import requests
import json
import os
import sys
import time
from datetime import datetime
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# WINDOWS FIX - Bypass Spark file operations
# --------------------------
def setup_windows_environment():
    """
    Setup Windows environment to avoid Hadoop native library issues
    We'll use Pandas instead of Spark file writes
    """
    import platform
    
    if platform.system() == "Windows":
        print("🔧 Windows OS detected - Using Pandas-based approach for compatibility...")
        
        # Set dummy HADOOP_HOME to suppress warnings
        hadoop_home = os.path.join(os.getcwd(), "hadoop")
        os.makedirs(hadoop_home, exist_ok=True)
        os.environ["HADOOP_HOME"] = hadoop_home
        return True
    return False

# Check if Windows
IS_WINDOWS = setup_windows_environment()

# Import PySpark after environment setup
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf, concat_ws
from pyspark.sql.types import FloatType, StringType

# --------------------------
# CONFIG
# --------------------------
GNEWS_API_KEY = '380af545145de6a25107d08ac9c4ac9c'
QUERY = 'stock market'
RAW_JSON_PATH = "news_data.json"
SENTIMENT_OUTPUT_DIR = "sentiment_output"
SENTIMENT_CSV_PATH = "sentiment_results.csv"
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
# STEP 2 & 3: Process with Pandas (Windows-friendly)
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
    if score is None or pd.isna(score):
        return "Neutral"
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def process_with_pandas():
    """
    Process news data using Pandas (more Windows-friendly)
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing News Data with Pandas...")
    
    if not os.path.exists(RAW_JSON_PATH):
        print("⚠️ No raw data found. Run fetch_news() first.")
        return None
    
    try:
        # Read JSON data
        articles = []
        with open(RAW_JSON_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    articles.append(json.loads(line))
                except:
                    continue
        
        if not articles:
            print("⚠️ No valid articles found.")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(articles)
        
        # Extract required columns
        required_cols = ['title', 'description', 'url', 'publishedAt', 'fetched_at']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        df = df[required_cols].copy()
        
        # Clean text
        df['text_clean'] = df['description'].fillna('').str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
        df['title_clean'] = df['title'].fillna('').str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
        
        # Filter out empty descriptions
        df = df[df['text_clean'].str.strip() != ''].copy()
        
        if len(df) == 0:
            print("⚠️ No valid articles after filtering.")
            return None
        
        print(f"✅ Processed {len(df)} articles")
        return df
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_sentiment_analysis_pandas(df):
    """
    Perform sentiment analysis using Pandas
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running Sentiment Analysis...")
    
    if df is None or len(df) == 0:
        print("⚠️ No data to analyze.")
        return None
    
    try:
        # Combine title and description
        df['combined_text'] = df['title_clean'] + ' ' + df['text_clean']
        
        # Calculate sentiment
        print("📊 Calculating sentiment scores...")
        df['polarity'] = df['combined_text'].apply(get_sentiment)
        df['sentiment'] = df['polarity'].apply(classify_sentiment)
        
        # Select final columns
        result_df = df[['title', 'description', 'url', 'polarity', 'sentiment', 'publishedAt', 'fetched_at']].copy()
        
        # Save results
        os.makedirs(SENTIMENT_OUTPUT_DIR, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(SENTIMENT_OUTPUT_DIR, 'sentiment_results.csv')
        result_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Saved results to '{csv_path}'")
        
        # Save as JSON for dashboard
        json_path = os.path.join(SENTIMENT_OUTPUT_DIR, 'consolidated.json')
        result_df.to_json(json_path, orient='records', lines=True, force_ascii=False)
        print(f"✅ Saved JSON to '{json_path}'")
        
        # Display sample
        print(f"\n📊 Sample Results:")
        print(result_df[['title', 'polarity', 'sentiment']].head(10).to_string(max_colwidth=60))
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

# --------------------------
# STEP 2 & 3: Process with PySpark (for non-Windows or if Spark works)
# --------------------------
def process_with_spark(spark):
    """
    Process news data using PySpark (fallback if Pandas fails)
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing with PySpark...")
    
    if not os.path.exists(RAW_JSON_PATH):
        print("⚠️ No raw data found.")
        return None
    
    try:
        # Read JSON
        df = spark.read.json(RAW_JSON_PATH)
        
        # Select and clean
        df = df.select(
            "title", "description", "url", "publishedAt", "fetched_at"
        ).withColumn(
            "text_clean",
            lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", ""))
        ).withColumn(
            "title_clean",
            lower(regexp_replace(col("title"), "[^a-zA-Z\\s]", ""))
        )
        
        # Filter empty
        df = df.filter(col("text_clean").isNotNull() & (col("text_clean") != ""))
        
        # Define UDFs
        sentiment_udf = udf(get_sentiment, FloatType())
        label_udf = udf(classify_sentiment, StringType())
        
        # Combine text using concat_ws (proper PySpark function)
        df = df.withColumn(
            "combined_text",
            concat_ws(" ", col("title_clean"), col("text_clean"))
        )
        
        # Calculate sentiment
        df = df.withColumn("polarity", sentiment_udf(col("combined_text")))
        df = df.withColumn("sentiment", label_udf(col("polarity")))
        
        # Select final columns
        result_df = df.select(
            "title", "description", "url", "polarity", "sentiment", 
            "publishedAt", "fetched_at"
        )
        
        # Convert to Pandas and save
        pandas_df = result_df.toPandas()
        
        os.makedirs(SENTIMENT_OUTPUT_DIR, exist_ok=True)
        
        csv_path = os.path.join(SENTIMENT_OUTPUT_DIR, 'sentiment_results.csv')
        pandas_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Saved results to '{csv_path}'")
        
        json_path = os.path.join(SENTIMENT_OUTPUT_DIR, 'consolidated.json')
        pandas_df.to_json(json_path, orient='records', lines=True, force_ascii=False)
        print(f"✅ Saved JSON to '{json_path}'")
        
        print(f"\n📊 Sample Results:")
        print(pandas_df[['title', 'polarity', 'sentiment']].head(10).to_string(max_colwidth=60))
        
        return pandas_df
        
    except Exception as e:
        print(f"❌ Error with PySpark processing: {e}")
        return None

# --------------------------
# STEP 4: Summary Statistics
# --------------------------
def print_sentiment_summary(df):
    """Print sentiment distribution summary"""
    if df is None or len(df) == 0:
        print("⚠️ No data available for summary.")
        return
    
    print(f"\n📊 Sentiment Summary [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:")
    print("=" * 50)
    
    try:
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{sentiment:>10}: {count:>4} articles ({percentage:>5.1f}%)")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Error generating summary: {e}")

# --------------------------
# STEP 5: Visualization
# --------------------------
def plot_sentiment_pie(df):
    """Generate pie chart of sentiment distribution"""
    if df is None or len(df) == 0:
        print("⚠️ No data available for plotting.")
        return
    
    print(f"\n📈 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating Pie Chart...")
    
    try:
        sentiment_counts = df['sentiment'].value_counts()
        
        labels = []
        sizes = []
        colors_map = {"Positive": "#66BC6A", "Neutral": "#FFA726", "Negative": "#EF5350"}
        colors = []
        
        for sentiment in ["Positive", "Neutral", "Negative"]:
            count = sentiment_counts.get(sentiment, 0)
            if count > 0:
                labels.append(sentiment)
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
            
            # Try to show (may not work in some environments)
            try:
                plt.show(block=False)
                plt.pause(0.1)
            except:
                pass
        else:
            print("⚠️ No data to plot.")
    except Exception as e:
        print(f"❌ Error creating plot: {e}")

# --------------------------
# CONTINUOUS MODE
# --------------------------
def run_continuous_pipeline(iterations=None):
    """
    Run the pipeline continuously
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
        
        # Fetch articles
        articles_count = fetch_news(append_mode=False)
        
        if articles_count > 0:
            # Process using Pandas (Windows-friendly)
            df = process_with_pandas()
            if df is not None:
                result_df = run_sentiment_analysis_pandas(df)
                if result_df is not None:
                    print_sentiment_summary(result_df)
        
        if iterations is None or iteration < iterations:
            print(f"\n⏳ Waiting {FETCH_INTERVAL} seconds before next fetch...")
            time.sleep(FETCH_INTERVAL)

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📰 Real-Time News Sentiment Analysis Pipeline")
    print("="*60)
    
    # Determine mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    
    try:
        if mode == "continuous":
            # Run continuously
            iterations = int(sys.argv[2]) if len(sys.argv) > 2 else None
            run_continuous_pipeline(iterations=iterations)
        else:
            # Run once
            print("\n🔹 Running pipeline once...")
            print(f"🔹 Processing mode: {'Pandas (Windows-compatible)' if IS_WINDOWS else 'Pandas'}")
            
            # Fetch news
            fetch_news()
            
            # Process with Pandas (works on all platforms)
            df = process_with_pandas()
            
            if df is not None:
                # Run sentiment analysis
                result_df = run_sentiment_analysis_pandas(df)
                
                if result_df is not None:
                    # Print summary
                    print_sentiment_summary(result_df)
                    
                    # Generate plot
                    plot_sentiment_pie(result_df)
                    
                    print("\n✅ Pipeline completed successfully!")
                    print(f"📂 Results saved in '{SENTIMENT_OUTPUT_DIR}' directory")
            else:
                print("\n⚠️ No data to process.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Pipeline finished.")