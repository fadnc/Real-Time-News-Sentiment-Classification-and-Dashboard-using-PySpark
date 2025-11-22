"""
Spark Structured Streaming Pipeline for Real-time News Sentiment Analysis

This script continuously monitors a directory for new JSON files containing news articles,
processes them in micro-batches, performs sentiment analysis, and writes results to output.
"""

import os
import time
from datetime import datetime
from textblob import TextBlob
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, udf, current_timestamp, 
    lit, concat_ws, when
)
from pyspark.sql.types import FloatType, StringType, StructType, StructField

# --------------------------
# CONFIG
# --------------------------
STREAMING_INPUT_DIR = "streaming_input"  # Directory to watch for new files
STREAMING_OUTPUT_DIR = "streaming_output"  # Output directory for results
CHECKPOINT_DIR = "checkpoint"  # Checkpoint directory for fault tolerance

# Create directories if they don't exist
for directory in [STREAMING_INPUT_DIR, STREAMING_OUTPUT_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# --------------------------
# SCHEMA DEFINITION
# --------------------------
news_schema = StructType([
    StructField("id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("description", StringType(), True),
    StructField("content", StringType(), True),
    StructField("url", StringType(), True),
    StructField("image", StringType(), True),
    StructField("publishedAt", StringType(), True),
    StructField("lang", StringType(), True),
    StructField("fetched_at", StringType(), True)
])

# --------------------------
# UDFs for Sentiment Analysis
# --------------------------
def get_sentiment_score(text):
    """Calculate sentiment polarity using TextBlob"""
    if text and isinstance(text, str) and len(text.strip()) > 0:
        try:
            return float(TextBlob(text).sentiment.polarity)
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

# Register UDFs
sentiment_score_udf = udf(get_sentiment_score, FloatType())
sentiment_label_udf = udf(classify_sentiment, StringType())

# --------------------------
# PROCESSING FUNCTIONS
# --------------------------
def process_batch(batch_df, batch_id):
    """
    Custom processing function for each micro-batch
    This is called for every batch of data
    """
    print(f"\n{'='*60}")
    print(f"Processing Batch #{batch_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Batch size: {batch_df.count()} records")
    print(f"{'='*60}")
    
    if batch_df.isEmpty():
        print("Batch is empty, skipping...")
        return
    
    # Show sample of incoming data
    print("\nSample of incoming articles:")
    batch_df.select("title", "description").show(3, truncate=60)
    
    # Show sentiment distribution
    print("\nSentiment distribution in this batch:")
    batch_df.groupBy("sentiment").count().show()

def create_streaming_pipeline(spark):
    """
    Create and configure the Spark Structured Streaming pipeline
    """
    print("\n🚀 Starting Spark Structured Streaming Pipeline")
    print(f"📂 Watching directory: {STREAMING_INPUT_DIR}")
    print(f"📂 Output directory: {STREAMING_OUTPUT_DIR}")
    print(f"📂 Checkpoint directory: {CHECKPOINT_DIR}\n")
    
    # Read streaming data from JSON files
    streaming_df = spark \
        .readStream \
        .schema(news_schema) \
        .json(STREAMING_INPUT_DIR) \
        .withColumn("processing_time", current_timestamp())
    
    # Data cleaning and preprocessing
    cleaned_df = streaming_df.select(
        "id",
        "title",
        "description",
        "url",
        "publishedAt",
        "fetched_at",
        "processing_time"
    ).withColumn(
        "text_clean",
        lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", ""))
    ).withColumn(
        "title_clean",
        lower(regexp_replace(col("title"), "[^a-zA-Z\\s]", ""))
    )
    
    # Filter out null or empty descriptions
    cleaned_df = cleaned_df.filter(
        col("text_clean").isNotNull() & (col("text_clean") != "")
    )
    
    # Sentiment analysis
    sentiment_df = cleaned_df.withColumn(
        "combined_text",
        concat_ws(" ", col("title_clean"), col("text_clean"))
    ).withColumn(
        "polarity",
        sentiment_score_udf(col("combined_text"))
    ).withColumn(
        "sentiment",
        sentiment_label_udf(col("polarity"))
    ).withColumn(
        "sentiment_confidence",
        when(col("polarity") > 0.5, "High")
        .when(col("polarity") < -0.5, "High")
        .when((col("polarity") > 0.2) | (col("polarity") < -0.2), "Medium")
        .otherwise("Low")
    )
    
    # Select final columns
    final_df = sentiment_df.select(
        "id",
        "title",
        "description",
        "url",
        "polarity",
        "sentiment",
        "sentiment_confidence",
        "publishedAt",
        "fetched_at",
        "processing_time"
    )
    
    return final_df

# --------------------------
# MAIN STREAMING APPLICATION
# --------------------------
def run_streaming_pipeline():
    """
    Main function to run the streaming pipeline
    """
    # Create Spark session with streaming configurations
    spark = SparkSession.builder \
        .appName("NewsSegmentStreamingPipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.streaming.schemaInference", "true") \
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*60)
    print("📊 Real-time News Sentiment Analysis - Streaming Pipeline")
    print("="*60)
    
    try:
        # Create the streaming pipeline
        final_df = create_streaming_pipeline(spark)
        
        # Write stream to console (for monitoring)
        console_query = final_df \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .option("numRows", 5) \
            .trigger(processingTime="10 seconds") \
            .start()
        
        # Write stream to JSON files (for dashboard)
        file_query = final_df \
            .writeStream \
            .outputMode("append") \
            .format("json") \
            .option("path", STREAMING_OUTPUT_DIR) \
            .option("checkpointLocation", f"{CHECKPOINT_DIR}/file") \
            .trigger(processingTime="10 seconds") \
            .start()
        
        # Write stream to in-memory table for queries
        memory_query = final_df \
            .writeStream \
            .outputMode("append") \
            .format("memory") \
            .queryName("news_sentiment") \
            .trigger(processingTime="10 seconds") \
            .start()
        
        # Custom processing with foreachBatch
        batch_query = final_df \
            .writeStream \
            .outputMode("append") \
            .foreachBatch(process_batch) \
            .trigger(processingTime="10 seconds") \
            .start()
        
        print("\n✅ Streaming queries started successfully!")
        print(f"\n📊 You can now:")
        print(f"   1. Drop JSON files into '{STREAMING_INPUT_DIR}' directory")
        print(f"   2. View results in '{STREAMING_OUTPUT_DIR}' directory")
        print(f"   3. Monitor the console output below")
        print(f"   4. Run the Streamlit dashboard to visualize results")
        print(f"\n⏹️  Press Ctrl+C to stop the streaming pipeline\n")
        
        # Wait for termination
        console_query.awaitTermination()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Streaming pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in streaming pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Stopping streaming queries...")
        spark.streams.awaitAnyTermination()
        spark.stop()
        print("✅ Spark session closed")

# --------------------------
# HELPER: Data Generator
# --------------------------
def generate_sample_stream(interval=10, duration=60):
    """
    Helper function to generate sample streaming data for testing
    Simulates continuous arrival of news articles
    """
    import json
    import requests
    from datetime import datetime
    
    GNEWS_API_KEY = '380af545145de6a25107d08ac9c4ac9c'
    
    print(f"\n🔄 Starting sample data generator")
    print(f"   Interval: {interval} seconds")
    print(f"   Duration: {duration} seconds\n")
    
    start_time = time.time()
    batch_num = 0
    
    while (time.time() - start_time) < duration:
        batch_num += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating batch #{batch_num}...")
        
        try:
            # Fetch news
            url = f"https://gnews.io/api/v4/search?q=stock+market&lang=en&max=10&token={GNEWS_API_KEY}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                # Write to streaming input directory
                timestamp = int(time.time())
                filename = f"{STREAMING_INPUT_DIR}/batch_{batch_num}_{timestamp}.json"
                
                with open(filename, "w", encoding="utf-8") as f:
                    for article in articles:
                        article['fetched_at'] = datetime.now().isoformat()
                        f.write(json.dumps(article) + "\n")
                
                print(f"   ✅ Written {len(articles)} articles to {filename}")
            else:
                print(f"   ❌ API error: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print(f"   ⏳ Waiting {interval} seconds...\n")
        time.sleep(interval)
    
    print(f"\n✅ Sample data generation complete ({batch_num} batches)")

# --------------------------
# ENTRY POINT
# --------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Run data generator for testing
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        generate_sample_stream(interval, duration)
    else:
        # Run streaming pipeline
        run_streaming_pipeline()