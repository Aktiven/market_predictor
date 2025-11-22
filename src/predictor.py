import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os


class MarketPredictor:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load trained models"""
        try:
            # Load classical ML models
            ml_models = ['random_forest', 'gradient_boosting', 'svm', 'neural_network']
            for model_name in ml_models:
                model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)

            # Load LSTM model
            lstm_path = os.path.join(self.models_dir, 'lstm_model.h5')
            if os.path.exists(lstm_path):
                self.models['lstm'] = load_model(lstm_path)

        except Exception as e:
            print(f"Error loading models: {e}")

    def predict(self, features, model_type='ensemble'):
        """Make predictions using specified model"""
        if not self.models:
            raise ValueError("No models loaded")

        if model_type == 'ensemble':
            return self.ensemble_predict(features)
        elif model_type in self.models:
            if model_type == 'lstm':
                # Reshape for LSTM
                if len(features.shape) == 2:
                    features = features.reshape((features.shape[0], features.shape[1], 1))
                return self.models[model_type].predict(features)
            else:
                return self.models[model_type].predict_proba(features)[:, 1]
        else:
            raise ValueError(f"Model {model_type} not found")

    def ensemble_predict(self, features):
        """Make ensemble prediction"""
        predictions = []

        for model_name, model in self.models.items():
            try:
                if model_name == 'lstm':
                    if len(features.shape) == 2:
                        features_reshaped = features.reshape((features.shape[0], features.shape[1], 1))
                    else:
                        features_reshaped = features
                    pred = model.predict(features_reshaped).flatten()
                else:
                    pred = model.predict_proba(features)[:, 1]
                predictions.append(pred)
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue

        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return None

    def get_confidence_intervals(self, predictions, method='bootstrap', n_iterations=1000):
        """Calculate confidence intervals for predictions"""
        if method == 'bootstrap':
            # Simple bootstrap method
            n_samples = len(predictions)
            bootstrap_means = []

            for _ in range(n_iterations):
                sample = np.random.choice(predictions, size=n_samples, replace=True)
                bootstrap_means.append(np.mean(sample))

            lower = np.percentile(bootstrap_means, 2.5)
            upper = np.percentile(bootstrap_means, 97.5)

            return lower, upper

        return np.percentile(predictions, 2.5), np.percentile(predictions, 97.5)