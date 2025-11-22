import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os


class ModelTrainer:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = {}

    def train_classical_ml(self, X_train, X_test, y_train, y_test):
        """Train classical machine learning models"""
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }

        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            print(f"{name} Accuracy: {accuracy:.4f}")

            # Save model
            joblib.dump(model, os.path.join(self.models_dir, f'{name}_model.pkl'))

        self.models.update(results)
        return results

    def build_lstm_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),

            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),

            LSTM(32, return_sequences=False),
            Dropout(0.3),

            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_lstm(self, X_train, X_test, y_train, y_test):
        """Train LSTM model"""
        # Reshape for LSTM
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5)
        ]

        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        print(f"LSTM Train Accuracy: {train_acc:.4f}")
        print(f"LSTM Test Accuracy: {test_acc:.4f}")

        # Save model
        model.save(os.path.join(self.models_dir, 'lstm_model.h5'))

        self.models['lstm'] = {
            'model': model,
            'accuracy': test_acc,
            'history': history
        }

        return self.models['lstm']

    def ensemble_prediction(self, X, models_list=None):
        """Make ensemble predictions"""
        if models_list is None:
            models_list = ['random_forest', 'gradient_boosting', 'lstm']

        predictions = []
        for model_name in models_list:
            if model_name in self.models:
                if model_name == 'lstm':
                    # Reshape for LSTM if needed
                    if len(X.shape) == 2:
                        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
                    else:
                        X_reshaped = X
                    pred = self.models[model_name]['model'].predict(X_reshaped)
                else:
                    pred = self.models[model_name]['model'].predict_proba(X)[:, 1]
                predictions.append(pred)

        if predictions:
            # Average predictions
            ensemble_pred = np.mean(predictions, axis=0)
            return ensemble_pred
        else:
            return None