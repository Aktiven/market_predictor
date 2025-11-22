import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class DataLoader:
    def __init__(self):
        self.data = {}

    def download_data(self, ticker, start_date, end_date):
        """Download historical market data"""
        try:
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not stock.empty:
                self.data[ticker] = stock
                print(f"Downloaded data for {ticker}")
            return stock
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            return None

    def load_multiple_tickers(self, tickers, start_date, end_date):
        """Load data for multiple tickers"""
        for ticker in tickers:
            self.download_data(ticker, start_date, end_date)
        return self.data

    def get_technical_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        df['bollinger_mid'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bollinger_high'] = df['bollinger_mid'] + (bb_std * 2)
        df['bollinger_low'] = df['bollinger_mid'] - (bb_std * 2)

        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # OBV
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        return df

    def prepare_features(self, ticker):
        """Prepare features for machine learning"""
        if ticker not in self.data:
            return None

        df = self.data[ticker].copy()
        df = self.get_technical_indicators(df)

        # Add price-based features
        df['price_change'] = df['Close'].pct_change()
        df['volume_change'] = df['Volume'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']

        # Add moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['Close'] / df[f'ma_{window}']

        # Target variable: Future price movement (1 if price goes up, 0 if down)
        df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

        # Drop NaN values
        df = df.dropna()

        return df

    def integrate_news_features(self, price_data: pd.DataFrame, news_data: pd.DataFrame) -> pd.DataFrame:
        """Integrate news and sentiment features with price data"""
        if news_data.empty:
            return price_data

        # Ensure dates are comparable
        price_data = price_data.copy()
        news_data = news_data.copy()
        news_data['date'] = pd.to_datetime(news_data['date']).dt.date
        price_data['date'] = price_data.index.date

        # Aggregate news by date
        daily_news = news_data.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'impact_score': 'mean',
            'event_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'no_news'
        }).round(4)

        # Flatten column names
        daily_news.columns = [f'news_{col[0]}_{col[1]}' if col[1] else f'news_{col[0]}'
                              for col in daily_news.columns]

        # Merge with price data
        merged_data = price_data.merge(daily_news, left_on='date', right_index=True, how='left')

        # Fill NaN values
        news_columns = [col for col in merged_data.columns if col.startswith('news_')]
        merged_data[news_columns] = merged_data[news_columns].fillna(0)
        merged_data['news_event_type'] = merged_data['news_event_type'].fillna('no_news')

        # Remove the temporary date column
        merged_data = merged_data.drop('date', axis=1)

        return merged_data

    def calculate_news_momentum(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate news sentiment momentum"""
        if 'news_sentiment_score_mean' in df.columns:
            df['news_sentiment_momentum'] = df['news_sentiment_score_mean'].rolling(window=window).mean()
            df['news_volume_momentum'] = df['news_sentiment_score_count'].rolling(window=window).mean()

        return df