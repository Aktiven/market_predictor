import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import yfinance as yf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Simple configuration
class Config:
    TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]
    START_DATE = "2020-01-01"
    SEQUENCE_LENGTH = 60

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
            return None
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            return None

class SimpleNewsCollector:
    """Simplified news collector for demonstration"""
    def __init__(self):
        self.news_templates = {
            'positive': [
                "Strong quarterly earnings beat analyst expectations",
                "New product launch drives investor optimism",
                "Strategic partnership announced to expand market share",
                "Positive regulatory developments boost confidence",
                "Company announces stock buyback program"
            ],
            'negative': [
                "Earnings miss estimates amid challenging market conditions",
                "Regulatory concerns weigh on stock performance",
                "Supply chain issues impact quarterly results",
                "Leadership changes create uncertainty",
                "Market volatility affects trading session"
            ],
            'neutral': [
                "Company holds annual shareholder meeting",
                "Regular trading session concludes",
                "Market awaits key economic data",
                "Sector analysis shows mixed performance",
                "Quarterly report filed with regulators"
            ]
        }
    
    def generate_news(self, ticker, start_date, end_date):
        """Generate simulated news data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        news_items = []
        
        for date in dates:
            if date.weekday() < 5:  # Only weekdays
                # Randomly decide if there's news (30% chance)
                if np.random.random() < 0.3:
                    sentiment = np.random.choice(['positive', 'negative', 'neutral'], 
                                               p=[0.4, 0.3, 0.3])
                    template = np.random.choice(self.news_templates[sentiment])
                    
                    # Generate sentiment score
                    if sentiment == 'positive':
                        sentiment_score = np.random.uniform(0.3, 0.9)
                    elif sentiment == 'negative':
                        sentiment_score = np.random.uniform(-0.9, -0.3)
                    else:
                        sentiment_score = np.random.uniform(-0.2, 0.2)
                    
                    news_items.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'title': f"{ticker}: {template}",
                        'summary': f"Detailed analysis of {ticker} {template.lower()}",
                        'source': 'Market News Network',
                        'sentiment_score': sentiment_score,
                        'impact_score': np.random.uniform(0.1, 0.8)
                    })
        
        return pd.DataFrame(news_items)

class MarketPredictorApp:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.news_collector = SimpleNewsCollector()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'news_loaded' not in st.session_state:
            st.session_state.news_loaded = False

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
        
        # Main content - Simplified tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Data Overview", 
            "📰 News Analysis",
            "🔮 Predictions"
        ])
        
        with tab1:
            self.show_data_overview(selected_ticker, start_date, end_date)
        
        with tab2:
            self.show_news_analysis(selected_ticker, start_date, end_date)
        
        with tab3:
            self.show_predictions(selected_ticker)

    def show_data_overview(self, ticker, start_date, end_date):
        st.header("Historical Data Overview")
        
        if st.button("Load Market Data") or st.session_state.data_loaded:
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
                        change_pct = (change / data['Close'][-2]) * 100
                        st.metric("Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
                    with col3:
                        st.metric("Volume", f"{data['Volume'][-1]:,}")
                    with col4:
                        volatility = data['Close'].pct_change().std() * 100
                        st.metric("Volatility", f"{volatility:.2f}%")
                    
                    # Show price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index, 
                        y=data['Close'],
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add moving averages
                    for window, color in [(20, 'orange'), (50, 'red')]:
                        ma = data['Close'].rolling(window=window).mean()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=ma,
                            name=f'MA {window}',
                            line=dict(color=color, width=1, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f'{ticker} Price Chart with Moving Averages',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show raw data
                    with st.expander("Show Raw Data"):
                        st.dataframe(data.tail(100))
                else:
                    st.error("Failed to load data. Please check your internet connection and try again.")

    def show_news_analysis(self, ticker, start_date, end_date):
        st.header("News Sentiment Analysis")
        
        if st.button("Generate News Data"):
            with st.spinner("Generating and analyzing news..."):
                try:
                    # Generate simulated news data
                    news_data = self.news_collector.generate_news(
                        ticker,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                    
                    if not news_data.empty:
                        st.session_state.news_data = news_data
                        st.session_state.news_loaded = True
                        
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
                        daily_sentiment = news_data.groupby('date')['sentiment_score'].mean()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=daily_sentiment.index,
                            y=daily_sentiment.values,
                            name='Daily Sentiment',
                            line=dict(color='purple', width=2),
                            fill='tozeroy'
                        ))
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(
                            title=f'{ticker} News Sentiment Over Time',
                            xaxis_title='Date',
                            yaxis_title='Sentiment Score',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Sentiment distribution
                        fig2 = px.histogram(
                            news_data, 
                            x='sentiment_score',
                            title='Sentiment Score Distribution',
                            nbins=20
                        )
                        fig2.add_vline(x=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Show recent news
                        st.subheader("Recent News Headlines")
                        for _, news in news_data.tail(10).iterrows():
                            sentiment = news['sentiment_score']
                            if sentiment > 0.1:
                                color = "🟢"  # Green
                            elif sentiment < -0.1:
                                color = "🔴"  # Red
                            else:
                                color = "⚪"  # Gray
                            
                            st.write(f"{color} **{news['date']}** - {news['title']}")
                            st.write(f"   Sentiment: `{sentiment:.3f}` | Impact: `{news['impact_score']:.3f}`")
                            st.write("---")
                    
                except Exception as e:
                    st.error(f"Error in news analysis: {str(e)}")
        else:
            if not st.session_state.get('news_loaded'):
                st.info("Click 'Generate News Data' to see news sentiment analysis")
            else:
                # Show existing news data
                news_data = st.session_state.news_data
                st.metric("Total News Articles", len(news_data))
                st.metric("Average Sentiment", f"{news_data['sentiment_score'].mean():.3f}")

    def show_predictions(self, ticker):
        st.header("Market Predictions")
        
        if not st.session_state.data_loaded:
            st.warning("Please load market data first in the Data Overview tab.")
            return
        
        st.info("""
        **Prediction Features Coming Soon:**
        - Machine learning price predictions
        - Technical analysis signals
        - News sentiment-based forecasts
        - Risk assessment metrics
        """)
        
        # Simple prediction based on recent trend
        data = st.session_state.data
        
        # Calculate simple metrics
        recent_return = (data['Close'][-1] - data['Close'][-5]) / data['Close'][-5] * 100
        volatility = data['Close'].pct_change().std() * 100
        volume_trend = data['Volume'][-5:].mean() / data['Volume'][-10:-5].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("5-Day Return", f"{recent_return:.2f}%")
        with col2:
            st.metric("Recent Volatility", f"{volatility:.2f}%")
        with col3:
            st.metric("Volume Trend", f"{volume_trend:.2f}x")
        
        # Simple prediction logic
        if recent_return > 2:
            prediction = "Bullish"
            confidence = min(80, 50 + abs(recent_return))
        elif recent_return < -2:
            prediction = "Bearish" 
            confidence = min(80, 50 + abs(recent_return))
        else:
            prediction = "Neutral"
            confidence = 50
        
        st.subheader("Simple Market Outlook")
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {confidence:.1f}%")
        
        # Explanation
        if prediction == "Bullish":
            st.success("Recent price action suggests positive momentum. Consider monitoring for entry opportunities.")
        elif prediction == "Bearish":
            st.warning("Recent declines indicate potential downward pressure. Exercise caution with new positions.")
        else:
            st.info("Market shows neutral characteristics. Wait for clearer directional signals.")

def main():
    try:
        app = MarketPredictorApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check the console for detailed error messages.")

if __name__ == "__main__":
    main()
