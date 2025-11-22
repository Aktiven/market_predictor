import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from typing import Dict, Tuple, List


class ImpactAnalyzer:
    def __init__(self):
        self.impact_models = {}
        self.historical_impact_db = self._initialize_impact_database()

    def _initialize_impact_database(self) -> pd.DataFrame:
        """Initialize database of historical event impacts"""
        # This would typically be loaded from a file or database
        return pd.DataFrame(columns=[
            'event_type', 'sector', 'market_cap', 'sentiment',
            'actual_impact', 'volatility_change', 'recovery_days'
        ])

    def analyze_historical_impact(self, events_with_price: pd.DataFrame) -> Dict:
        """Analyze historical impact patterns"""
        if events_with_price.empty:
            return {}

        analysis = {}

        # Impact by event type
        event_impact = events_with_price.groupby('event_type').agg({
            'actual_price_change': ['mean', 'std', 'count'],
            'actual_impact': 'mean',
            'prediction_accuracy': 'mean'
        }).round(4)

        analysis['event_type_impact'] = event_impact

        # Sentiment vs actual impact correlation
        sentiment_corr = events_with_price['sentiment'].corr(events_with_price['actual_price_change'])
        analysis['sentiment_correlation'] = sentiment_corr

        # Impact persistence analysis
        analysis['impact_persistence'] = self._analyze_impact_persistence(events_with_price)

        return analysis

    def _analyze_impact_persistence(self, events_df: pd.DataFrame) -> Dict:
        """Analyze how long event impacts persist"""
        persistence_analysis = {}

        for event_type in events_df['event_type'].unique():
            type_events = events_df[events_df['event_type'] == event_type]
            if len(type_events) > 5:  # Minimum events for analysis
                # Calculate how many days until price returns to pre-event level
                # This is simplified - in practice you'd need more detailed price data
                positive_events = type_events[type_events['actual_price_change'] > 0]
                negative_events = type_events[type_events['actual_price_change'] < 0]

                persistence_analysis[event_type] = {
                    'avg_positive_impact_duration': len(positive_events),
                    'avg_negative_impact_duration': len(negative_events),
                    'total_events': len(type_events)
                }

        return persistence_analysis

    def train_impact_prediction_model(self, historical_events: pd.DataFrame) -> RandomForestRegressor:
        """Train model to predict impact of future events"""
        if historical_events.empty:
            return None

        # Prepare features
        feature_columns = ['sentiment', 'impact_score', 'confidence']
        if 'sector_encoded' in historical_events.columns:
            feature_columns.append('sector_encoded')

        X = historical_events[feature_columns]
        y = historical_events['actual_impact']

        # Handle categorical variables
        if 'sector' in historical_events.columns and 'sector_encoded' not in historical_events.columns:
            X = pd.get_dummies(historical_events, columns=['sector'])
            feature_columns = [col for col in X.columns if col != 'actual_impact']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Impact Prediction Model - MAE: {mae:.4f}, R²: {r2:.4f}")

        self.impact_models['random_forest'] = model
        return model

    def predict_event_impact(self, event_features: Dict) -> Dict[str, float]:
        """Predict impact of a new event"""
        if 'random_forest' not in self.impact_models:
            return self._fallback_impact_prediction(event_features)

        try:
            # Prepare features for prediction
            feature_vector = self._prepare_feature_vector(event_features)
            predicted_impact = self.impact_models['random_forest'].predict([feature_vector])[0]

            return {
                'predicted_impact': max(0, predicted_impact),
                'confidence': 0.7,  # Model confidence
                'expected_direction': 1 if event_features.get('sentiment', 0) > 0 else -1,
                'volatility_effect': predicted_impact * 0.3  # Estimated volatility multiplier
            }
        except Exception as e:
            print(f"Impact prediction error: {e}")
            return self._fallback_impact_prediction(event_features)

    def _prepare_feature_vector(self, event_features: Dict) -> List[float]:
        """Prepare feature vector for model prediction"""
        # This would need to match the training feature structure
        features = []
        features.append(event_features.get('sentiment', 0))
        features.append(event_features.get('impact_score', 0.5))
        features.append(event_features.get('confidence', 0.5))

        # Add sector encoding if available
        if 'sector_encoded' in event_features:
            features.append(event_features['sector_encoded'])

        return features

    def _fallback_impact_prediction(self, event_features: Dict) -> Dict[str, float]:
        """Fallback impact prediction using rule-based approach"""
        event_type = event_features.get('event_type', 'unknown')
        sentiment = event_features.get('sentiment', 0)

        base_impacts = {
            'earnings': 0.02,
            'mergers_acquisitions': 0.03,
            'regulatory': 0.04,
            'product_news': 0.015,
            'leadership': 0.008,
            'economic': 0.012,
            'unknown': 0.01
        }

        base_impact = base_impacts.get(event_type, 0.01)
        adjusted_impact = base_impact * (1 + abs(sentiment))

        return {
            'predicted_impact': adjusted_impact,
            'confidence': 0.5,
            'expected_direction': 1 if sentiment > 0 else -1,
            'volatility_effect': adjusted_impact * 0.4
        }

    def calculate_market_overreaction(self, events_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Identify events where market overreacted"""
        overreaction_events = []

        for _, event in events_df.iterrows():
            event_date = pd.to_datetime(event['date'])

            # Calculate short-term vs long-term impact
            short_term_change = self._get_price_change(price_data, event_date, 1, 3)  # 1-3 days after
            medium_term_change = self._get_price_change(price_data, event_date, 5, 10)  # 5-10 days after

            if short_term_change is not None and medium_term_change is not None:
                overreaction = abs(short_term_change) - abs(medium_term_change)

                overreaction_events.append({
                    **event,
                    'short_term_change': short_term_change,
                    'medium_term_change': medium_term_change,
                    'overreaction_score': max(0, overreaction),
                    'is_overreaction': overreaction > 0.02  # 2% threshold
                })

        return pd.DataFrame(overreaction_events)

    def _get_price_change(self, price_data: pd.DataFrame, event_date: datetime,
                          days_after: int, window: int) -> float:
        """Calculate price change over specified window"""
        start_date = event_date + timedelta(days=days_after)
        end_date = start_date + timedelta(days=window)

        start_prices = price_data[price_data.index >= start_date]
        end_prices = price_data[price_data.index >= end_date]

        if len(start_prices) > 0 and len(end_prices) > 0:
            start_price = start_prices.iloc[0]['Close']
            end_price = end_prices.iloc[0]['Close']
            return (end_price - start_price) / start_price

        return None