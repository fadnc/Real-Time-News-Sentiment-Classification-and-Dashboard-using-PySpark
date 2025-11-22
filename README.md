# Real-Time-News-Sentiment-Classification-and-Dashboard-using-PySpark

# Real-Time News Sentiment Classification and Dashboard using PySpark

A comprehensive real-time news sentiment analysis system that fetches news articles, classifies them using PySpark ML pipeline, and visualizes results in an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.x-orange)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io/)

## 🌟 Features

- **Real-time News Fetching**: Automatically fetches latest news from GNews API
- **Sentiment Analysis**: Uses TextBlob for polarity-based sentiment classification
- **PySpark Processing**: Scalable data processing with Apache Spark
- **Spark Structured Streaming**: Optional continuous streaming pipeline
- **Interactive Dashboard**: Real-time Streamlit dashboard with auto-refresh
- **Visualizations**: Pie charts, bar charts, and polarity distributions
- **Batch & Continuous Modes**: Run once or continuously monitor news

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Java 8 or 11 (for PySpark)
- 2GB+ RAM recommended

### Python Packages
```
streamlit
pyspark
textblob
requests
pandas
matplotlib
plotly
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fadnc-real-time-news-sentiment-classification-and-dashboard-using-pyspark.git
cd fadnc-real-time-news-sentiment-classification-and-dashboard-using-pyspark
```

### 2. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Install Java (if not already installed)
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install openjdk-11-jdk

# macOS:
brew install openjdk@11

# Download TextBlob corpora
python -m textblob.download_corpora
```

### 3. Configure API Key
Edit `work.py` and update your GNews API key:
```python
GNEWS_API_KEY = 'your_api_key_here'
```

Get your free API key from [GNews.io](https://gnews.io/)

## 📖 Usage

### Option 1: Basic Pipeline (Run Once)

Fetch news, analyze sentiment, and generate visualizations:

```bash
python work.py
```

This will:
1. Fetch latest news articles
2. Preprocess text data with PySpark
3. Perform sentiment analysis
4. Generate summary statistics
5. Create visualization chart

### Option 2: Continuous Pipeline

Run the pipeline continuously with automatic updates:

```bash
# Run indefinitely (Ctrl+C to stop)
python work.py continuous

# Run for 5 iterations
python work.py continuous 5
```

### Option 3: Spark Structured Streaming

For true real-time streaming using Spark Structured Streaming:

```bash
# Start the streaming pipeline
python streaming_pipeline.py

# In another terminal, generate sample streaming data
python streaming_pipeline.py generate 15 120
# (generates data every 15 seconds for 120 seconds)
```

### Launch the Dashboard

Start the Streamlit dashboard to visualize results:

```bash
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Dashboard Features

### Main Metrics
- Total article count
- Positive/Negative/Neutral distribution
- Average polarity score
- Last update timestamp

### Visualizations
1. **Sentiment Distribution Pie Chart**: Shows percentage breakdown
2. **Sentiment Counts Bar Chart**: Displays absolute counts
3. **Polarity Distribution Histogram**: Shows score distribution
4. **Recent Headlines**: Latest articles with sentiment labels

### Dashboard Controls
- **Auto-refresh**: Toggle automatic updates
- **Refresh Interval**: Adjust update frequency (5-60 seconds)
- **Manual Refresh**: Force immediate update
- **Display Options**: Show/hide raw data, adjust number of headlines

## 🗂️ Project Structure

```
fadnc-real-time-news-sentiment-classification-and-dashboard-using-pyspark/
├── README.md                   # Documentation
├── requirements.txt            # Python dependencies
├── packages.txt               # System packages
├── work.py                    # Main pipeline script
├── streamlit_app.py           # Streamlit dashboard
├── streaming_pipeline.py      # Spark Structured Streaming (optional)
├── news_data.json            # Raw news data
├── cleaned_news_output/      # Preprocessed data
├── sentiment_output/         # Sentiment analysis results
├── streaming_input/          # Streaming input directory
├── streaming_output/         # Streaming output directory
└── checkpoint/               # Streaming checkpoints
```

## 🔧 Configuration

### Customize News Query
Edit `work.py`:
```python
QUERY = 'stock market'  # Change to any topic
```

### Adjust Fetch Interval
```python
FETCH_INTERVAL = 300  # Seconds between fetches (default: 5 minutes)
```

### Sentiment Classification Thresholds
```python
def classify_sentiment(score):
    if score > 0.1:      # Positive threshold
        return "Positive"
    elif score < -0.1:   # Negative threshold
        return "Negative"
    else:
        return "Neutral"
```

## 📈 Example Output

### Console Output
```
[2025-09-29 10:30:15] Fetching News from GNews API...
✅ Fetched and saved 50 articles.

[2025-09-29 10:30:20] Preprocessing News Data with PySpark...
✅ Saved cleaned data to cleaned_news_output

[2025-09-29 10:30:25] Running Sentiment Analysis...
✅ Saved sentiment results to 'sentiment_output'

📊 Sentiment Summary:
==================================================
  Positive:   28 articles ( 56.0%)
   Neutral:   15 articles ( 30.0%)
  Negative:    7 articles ( 14.0%)
==================================================
```

### Sample Sentiment Results
| Title | Polarity | Sentiment |
|-------|----------|-----------|
| Stock Market Reaches New Highs | 0.352 | Positive |
| Mixed Signals in Trading Session | 0.023 | Neutral |
| Investors Concerned About Economy | -0.245 | Negative |

## 🎯 Use Cases

1. **Financial News Monitoring**: Track market sentiment in real-time
2. **Brand Monitoring**: Monitor sentiment about your company
3. **Market Research**: Analyze public opinion on topics
4. **News Aggregation**: Classify and organize news by sentiment
5. **Academic Research**: Study sentiment trends over time

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `Java not found` error
```bash
# Solution: Install Java and set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
```

**Issue**: API rate limit exceeded
```bash
# Solution: Increase FETCH_INTERVAL or use a paid API plan
FETCH_INTERVAL = 600  # Fetch every 10 minutes
```

**Issue**: Dashboard shows no data
```bash
# Solution: Ensure work.py has run successfully
python work.py  # Run pipeline first
streamlit run streamlit_app.py  # Then start dashboard
```

**Issue**: Memory errors with large datasets
```bash
# Solution: Increase Spark driver memory
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

## 🔮 Future Enhancements

- [ ] Support for multiple news APIs (NewsAPI, Bing News, etc.)
- [ ] Advanced ML models (BERT, RoBERTa) for sentiment analysis
- [ ] Historical data analysis and trend tracking
- [ ] Export functionality (PDF reports, CSV exports)
- [ ] Email alerts for sentiment thresholds
- [ ] Multi-language support
- [ ] Integration with databases (PostgreSQL, MongoDB)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, Azure, GCP)

## 📝 API Information

This project uses the [GNews API](https://gnews.io/):
- Free tier: 100 requests/day
- Pro tier: Higher limits available
- Supports multiple languages
- Real-time news from 60,000+ sources

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Apache Spark for distributed processing
- TextBlob for sentiment analysis
- Streamlit for dashboard framework
- GNews.io for news API
- Plotly for interactive visualizations

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**