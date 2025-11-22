import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class MarketVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_price_history(self, df, ticker):
        """Plot historical price data with technical indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Price and Bollinger Bands', 'RSI', 'MACD'],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )

        # Price and Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bollinger_high'], name='BB High',
                       line=dict(color='red', dash='dash'), opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bollinger_low'], name='BB Low',
                       line=dict(color='green', dash='dash'), opacity=0.7),
            row=1, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )

        fig.update_layout(
            title=f'{ticker} Technical Analysis',
            height=800,
            showlegend=True
        )

        return fig

    def plot_predictions(self, actual, predictions, dates, confidence_intervals=None):
        """Plot actual vs predicted values"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=actual,
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=dates, y=predictions,
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))

        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            fig.add_trace(go.Scatter(
                x=dates, y=lower,
                name='Lower CI',
                line=dict(color='gray', width=1),
                opacity=0.3
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=upper,
                name='Upper CI',
                line=dict(color='gray', width=1),
                fill='tonexty',
                opacity=0.3
            ))

        fig.update_layout(
            title='Market Predictions vs Actual',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500
        )

        return fig

    def plot_feature_importance(self, feature_names, importance_scores):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)

        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance'
        )

        return fig

    def plot_model_comparison(self, model_results):
        """Compare model performance"""
        models = list(model_results.keys())
        accuracies = [result['accuracy'] for result in model_results.values()]

        fig = px.bar(
            x=models,
            y=accuracies,
            title='Model Performance Comparison',
            labels={'x': 'Model', 'y': 'Accuracy'},
            color=accuracies,
            color_continuous_scale='Viridis'
        )

        return fig