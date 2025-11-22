import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None

    def create_sequences(self, data, sequence_length):
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i - sequence_length:i])
            y.append(data[i, 0])  # Assuming close price is the first column
        return np.array(X), np.array(y)

    def prepare_lstm_features(self, df, sequence_length=60):
        """Prepare features for LSTM model"""
        features = ['Close', 'Volume', 'rsi', 'macd', 'bollinger_high',
                    'bollinger_low', 'atr', 'obv']

        # Select and scale features
        feature_data = df[features].values
        scaled_data = self.scaler.fit_transform(feature_data)

        # Create sequences
        X, y = self.create_sequences(scaled_data, sequence_length)

        return X, y, self.scaler

    def select_features(self, X, y, k=20):
        """Select most important features"""
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        return X_selected

    def add_technical_features(self, df):
        """Add advanced technical features"""
        # Price momentum
        df['momentum'] = df['Close'] / df['Close'].shift(10) - 1

        # Volatility
        df['volatility'] = df['Close'].pct_change().rolling(20).std()

        # Price position in Bollinger Bands
        df['bb_position'] = (df['Close'] - df['bollinger_low']) / (df['bollinger_high'] - df['bollinger_low'])

        # RSI signals
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

        return df