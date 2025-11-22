import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import MarketPredictor
from visualizer import MarketVisualizer


class MarketPredictorApp:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.visualizer = MarketVisualizer()
        self.predictor = MarketPredictor(self.config.MODELS_DIR)

        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False

    def run(self):
        st.set_page_config(
            page_title="AI Market Predictor",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🤖 AI-Powered Historical Market Predictor")
        st.markdown("""
        This application uses machine learning and historical data to predict market movements.
        **Disclaimer:** This is for educational purposes only. Past performance doesn't guarantee future results.
        """)

        # Sidebar
        st.sidebar.title("Configuration")

        # Ticker selection
        selected_ticker = st.sidebar.selectbox(
            "Select Ticker",
            self.config.TICKERS,
            index=0
        )

        # Date range
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.strptime(self.config.START_DATE, "%Y-%m-%d")
            )
        with col2:
            end_date = st.date_input("End Date", datetime.now())

        # Model selection
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["ensemble", "lstm", "random_forest", "gradient_boosting"],
            index=0
        )

        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview",
            "🤖 Model Training",
            "🔮 Predictions",
            "📈 Analysis"
        ])

        with tab1:
            self.show_data_overview(selected_ticker, start_date, end_date)

        with tab2:
            self.show_model_training(selected_ticker)

        with tab3:
            self.show_predictions(selected_ticker, model_type)

        with tab4:
            self.show_analysis(selected_ticker)

    def show_data_overview(self, ticker, start_date, end_date):
        st.header("Historical Data Overview")

        if st.button("Load Data") or st.session_state.data_loaded:
            with st.spinner("Loading market data..."):
                data = self.data_loader.download_data(
                    ticker,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )

                if data is not None:
                    st.session_state.data_loaded = True
                    st.session_state.data = data
                    st.session_state.ticker = ticker

                    # Display basic statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${data['Close'][-1]:.2f}")
                    with col2:
                        change = data['Close'][-1] - data['Close'][-2]
                        st.metric("Daily Change", f"${change:.2f}")
                    with col3:
                        st.metric("Volume", f"{data['Volume'][-1]:,}")
                    with col4:
                        volatility = data['Close'].pct_change().std()
                        st.metric("Volatility", f"{volatility:.4f}")

                    # Show price chart
                    fig = self.visualizer.plot_price_history(data, ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    # Show raw data
                    with st.expander("Show Raw Data"):
                        st.dataframe(data.tail(100))
                else:
                    st.error("Failed to load data. Please check your internet connection and try again.")

    def show_model_training(self, ticker):
        st.header("Model Training")

        if not st.session_state.data_loaded:
            st.warning("Please load data first in the Data Overview tab.")
            return

        if st.button("Train Models"):
            with st.spinner("Training machine learning models..."):
                try:
                    # Prepare features
                    df = self.data_loader.prepare_features(ticker)

                    # Prepare features for classical ML
                    features = [col for col in df.columns if
                                col not in ['target', 'Close', 'High', 'Low', 'Open', 'Volume']]
                    X = df[features].values
                    y = df['target'].values

                    # Split data
                    split_idx = int(len(X) * self.config.TRAIN_TEST_SPLIT)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    # Train models
                    trainer = ModelTrainer(self.config.MODELS_DIR)
                    ml_results = trainer.train_classical_ml(X_train, X_test, y_train, y_test)

                    # Prepare LSTM features
                    X_lstm, y_lstm, scaler = self.feature_engineer.prepare_lstm_features(
                        df, self.config.SEQUENCE_LENGTH
                    )

                    split_idx_lstm = int(len(X_lstm) * self.config.TRAIN_TEST_SPLIT)
                    X_train_lstm, X_test_lstm = X_lstm[:split_idx_lstm], X_lstm[split_idx_lstm:]
                    y_train_lstm, y_test_lstm = y_lstm[:split_idx_lstm], y_lstm[split_idx_lstm:]

                    lstm_results = trainer.train_lstm(
                        X_train_lstm, X_test_lstm,
                        y_train_lstm, y_test_lstm
                    )

                    st.session_state.models_trained = True
                    st.session_state.trainer = trainer
                    st.session_state.df = df

                    st.success("Models trained successfully!")

                    # Show model comparison
                    all_results = {**ml_results, 'lstm': lstm_results}
                    fig = self.visualizer.plot_model_comparison(all_results)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error training models: {e}")

    def show_predictions(self, ticker, model_type):
        st.header("Market Predictions")

        if not st.session_state.models_trained:
            st.warning("Please train models first in the Model Training tab.")
            return

        # Make predictions
        df = st.session_state.df
        trainer = st.session_state.trainer

        # Use recent data for prediction
        features = [col for col in df.columns if col not in ['target', 'Close', 'High', 'Low', 'Open', 'Volume']]
        X_recent = df[features].values[-100:]  # Last 100 days

        predictions = trainer.ensemble_prediction(X_recent)

        if predictions is not None:
            # Create prediction timeline
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, len(predictions) + 1)]

            # Plot predictions
            actual_prices = df['Close'].values[-100:]
            actual_dates = df.index[-100:]

            fig = self.visualizer.plot_predictions(
                actual_prices,
                predictions * actual_prices[-1],  # Scale predictions
                actual_dates
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show prediction metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                current_pred = predictions[-1]
                st.metric("Next Day Prediction",
                          "UP" if current_pred > 0.5 else "DOWN",
                          f"{(current_pred - 0.5) * 200:.1f}% confidence")
            with col2:
                avg_confidence = np.mean(np.abs(predictions - 0.5)) * 2
                st.metric("Average Confidence", f"{avg_confidence * 100:.1f}%")
            with col3:
                success_rate = np.mean((predictions > 0.5) == (df['target'].values[-100:] == 1))
                st.metric("Historical Accuracy", f"{success_rate * 100:.1f}%")

    def show_analysis(self, ticker):
        st.header("Market Analysis")

        if not st.session_state.data_loaded:
            st.warning("Please load data first in the Data Overview tab.")
            return

        df = st.session_state.data

        # Technical analysis
        st.subheader("Technical Indicators")

        col1, col2, col3 = st.columns(3)

        with col1:
            current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            st.metric("RSI", f"{current_rsi:.1f}")
            if current_rsi < 30:
                st.success("Oversold Territory")
            elif current_rsi > 70:
                st.warning("Overbought Territory")
            else:
                st.info("Neutral Territory")

        with col2:
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd_signal = "BULLISH" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "BEARISH"
                st.metric("MACD Signal", macd_signal)

        with col3:
            if 'bollinger_high' in df.columns and 'bollinger_low' in df.columns:
                current_price = df['Close'].iloc[-1]
                bb_position = (current_price - df['bollinger_low'].iloc[-1]) / \
                              (df['bollinger_high'].iloc[-1] - df['bollinger_low'].iloc[-1])
                st.metric("Bollinger Position", f"{bb_position * 100:.1f}%")

        # Volatility analysis
        st.subheader("Volatility Analysis")
        returns = df['Close'].pct_change().dropna()

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns'))
            fig.update_layout(title="Return Distribution", xaxis_title="Daily Returns", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            volatility = returns.rolling(20).std()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[20:], y=volatility[20:], name='20-day Volatility'))
            fig.update_layout(title="Historical Volatility", xaxis_title="Date", yaxis_title="Volatility")
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        # Existing setup...

        # Add new tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Overview",
            "🤖 Model Training",
            "🔮 Predictions",
            "📈 Analysis",
            "📰 News Analysis",
            "🎯 Event Impact"
        ])

        # Existing tabs...

        with tab5:
            self.show_news_analysis(selected_ticker, start_date, end_date)

        with tab6:
            self.show_event_impact(selected_ticker)

    def show_news_analysis(self, ticker, start_date, end_date):
        st.header("News Sentiment Analysis")

        if st.button("Collect News Data"):
            with st.spinner("Collecting and analyzing news..."):
                try:
                    from news_analyzer.news_collector import NewsCollector
                    from news_analyzer.sentiment_analyzer import SentimentAnalyzer

                    news_collector = NewsCollector()
                    sentiment_analyzer = SentimentAnalyzer()

                    # Collect news
                    news_data = news_collector.collect_historical_news(
                        ticker,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )

                    if not news_data.empty:
                        st.session_state.news_data = news_data

                        # Display news summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total News Articles", len(news_data))
                        with col2:
                            avg_sentiment = news_data['sentiment_score'].mean()
                            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                        with col3:
                            positive_news = len(news_data[news_data['sentiment_score'] > 0.1])
                            st.metric("Positive News", positive_news)

                        # Sentiment over time
                        news_data['date'] = pd.to_datetime(news_data['date'])
                        sentiment_timeline = news_data.groupby('date')['sentiment_score'].mean()

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=sentiment_timeline.index,
                            y=sentiment_timeline.values,
                            name='News Sentiment',
                            line=dict(color='purple', width=2)
                        ))
                        fig.update_layout(
                            title=f'{ticker} News Sentiment Over Time',
                            xaxis_title='Date',
                            yaxis_title='Sentiment Score'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Show recent news
                        with st.expander("Recent News Headlines"):
                            for _, news in news_data.tail(10).iterrows():
                                sentiment_color = "green" if news['sentiment_score'] > 0.1 else "red" if news[
                                                                                                             'sentiment_score'] < -0.1 else "gray"
                                st.write(f"**{news['date']}** - {news['title']}")
                                st.write(f"Sentiment: :{sentiment_color}[{news['sentiment_score']:.3f}]")
                                st.write("---")

                except Exception as e:
                    st.error(f"Error analyzing news: {e}")

    def show_event_impact(self, ticker):
        st.header("Event Impact Analysis")

        if not st.session_state.get('news_data') or not st.session_state.data_loaded:
            st.warning("Please load both price data and news data first.")
            return

        if st.button("Analyze Event Impact"):
            with st.spinner("Analyzing event impacts..."):
                try:
                    from news_analyzer.event_extractor import EventExtractor
                    from news_analyzer.impact_analyzer import ImpactAnalyzer

                    event_extractor = EventExtractor()
                    impact_analyzer = ImpactAnalyzer()

                    # Extract events
                    events_df = event_extractor.extract_events_from_news(st.session_state.news_data)

                    # Correlate with price data
                    correlated_events = event_extractor.correlate_events_with_price(
                        events_df, st.session_state.data
                    )

                    if not correlated_events.empty:
                        st.session_state.correlated_events = correlated_events

                        # Display impact analysis
                        st.subheader("Event Impact Summary")

                        # Impact by event type
                        impact_summary = correlated_events.groupby('event_type').agg({
                            'actual_price_change': ['mean', 'count'],
                            'prediction_accuracy': 'mean'
                        }).round(4)

                        st.dataframe(impact_summary)

                        # Train impact prediction model
                        model = impact_analyzer.train_impact_prediction_model(correlated_events)

                        if model is not None:
                            st.success("Impact prediction model trained successfully!")

                            # Show feature importance
                            feature_importance = pd.DataFrame({
                                'feature': ['sentiment', 'impact_score', 'confidence'],
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)

                            fig = px.bar(
                                feature_importance,
                                x='importance',
                                y='feature',
                                title='Event Impact Prediction Feature Importance'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Overreaction analysis
                        overreaction_events = impact_analyzer.calculate_market_overreaction(
                            correlated_events, st.session_state.data
                        )

                        if not overreaction_events.empty:
                            st.subheader("Market Overreaction Detection")
                            overreaction_count = len(overreaction_events[overreaction_events['is_overreaction']])
                            st.metric("Overreaction Events Detected", overreaction_count)

                            # Show overreaction examples
                            if overreaction_count > 0:
                                st.write("Top Overreaction Events:")
                                overreaction_examples = overreaction_events[
                                    overreaction_events['is_overreaction']
                                ].nlargest(5, 'overreaction_score')

                                for _, event in overreaction_examples.iterrows():
                                    st.write(f"**{event['date']}** - {event['event_type']}")
                                    st.write(f"Short-term: {event['short_term_change']:.3f}, "
                                             f"Medium-term: {event['medium_term_change']:.3f}")
                                    st.write(f"Overreaction Score: {event['overreaction_score']:.3f}")
                                    st.write("---")

                except Exception as e:
                    st.error(f"Error in event impact analysis: {e}")


if __name__ == "__main__":
    app = MarketPredictorApp()
    app.run()