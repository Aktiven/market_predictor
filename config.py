import os
from datetime import datetime, timedelta


class Config:
    # Data settings
    START_DATE = "2010-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]

    # Model settings
    TRAIN_TEST_SPLIT = 0.8
    SEQUENCE_LENGTH = 60  # days for time series prediction
    PREDICTION_DAYS = 30

    # Feature engineering
    TECHNICAL_INDICATORS = [
        'rsi', 'macd', 'bollinger_high', 'bollinger_low',
        'atr', 'obv', 'cci', 'williams_r'
    ]

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(DATA_DIR, "models")

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    # News and Events settings
    NEWS_SOURCES = ['alphavantage', 'financialmodelingprep', 'simulated']
    SENTIMENT_ANALYSIS_METHODS = ['vader', 'textblob', 'finbert']

    # Event impact analysis
    IMPACT_ANALYSIS_WINDOW = 5  # days to analyze impact
    OVERREACTION_THRESHOLD = 0.02  # 2% threshold for overreaction

    # News collection
    NEWS_COLLECTION_DAYS = 30  # How many days back to collect news

    # Paths
    NEWS_DATA_DIR = os.path.join(BASE_DIR, "data", "news_events")
    SENTIMENT_DATA_DIR = os.path.join(NEWS_DATA_DIR, "sentiment")

    # Ensure directories exist
    os.makedirs(NEWS_DATA_DIR, exist_ok=True)
    os.makedirs(SENTIMENT_DATA_DIR, exist_ok=True)