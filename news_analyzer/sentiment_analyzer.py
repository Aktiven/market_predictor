# news_analyzer/sentiment_analyzer_compatible.py
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List


class CompatibleSentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.financial_lexicon = self._load_financial_lexicon()

    def _load_financial_lexicon(self) -> Dict:
        """Load financial-specific sentiment words"""
        financial_positive = [
            'profit', 'gain', 'growth', 'beat', 'surge', 'rally', 'soar', 'jump',
            'increase', 'rise', 'up', 'bullish', 'optimistic', 'strong', 'record',
            'approval', 'acquisition', 'merger', 'dividend', 'buyback'
        ]

        financial_negative = [
            'loss', 'decline', 'fall', 'drop', 'plunge', 'crash', 'miss', 'cut',
            'reduce', 'down', 'bearish', 'pessimistic', 'weak', 'warning',
            'rejection', 'investigation', 'lawsuit', 'bankruptcy', 'default'
        ]

        return {
            'positive': {word: 1.5 for word in financial_positive},
            'negative': {word: -1.5 for word in financial_negative}
        }

    def analyze_news_sentiment(self, news_text: str) -> Dict[str, float]:
        """Comprehensive sentiment analysis using compatible methods"""
        if not news_text or len(news_text.strip()) < 10:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        methods = [
            self._vader_sentiment,
            self._textblob_sentiment,
            self._financial_sentiment
        ]

        sentiments = []
        for method in methods:
            try:
                sentiment = method(news_text)
                sentiments.append(sentiment)
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
                continue

        if not sentiments:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        return self._ensemble_sentiment(sentiments)

    def _vader_sentiment(self, text: str) -> Dict[str, float]:
        """VADER sentiment analysis"""
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'method': 'vader'
        }

    def _textblob_sentiment(self, text: str) -> Dict[str, float]:
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        compound = polarity
        positive = max(0, polarity)
        negative = max(0, -polarity)
        neutral = 1 - abs(polarity)

        return {
            'compound': compound,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'subjectivity': subjectivity,
            'method': 'textblob'
        }

    def _financial_sentiment(self, text: str) -> Dict[str, float]:
        """Financial-domain enhanced sentiment analysis"""
        # Start with VADER
        vader_scores = self.vader_sentiment(text)

        # Enhance with financial lexicon
        financial_score = self._calculate_financial_score(text)

        # Combine scores (weight financial lexicon)
        combined_compound = (vader_scores['compound'] + financial_score * 0.3) / 1.3

        return {
            'compound': combined_compound,
            'positive': max(0, combined_compound),
            'negative': max(0, -combined_compound),
            'neutral': 1 - abs(combined_compound),
            'method': 'financial'
        }

    def _calculate_financial_score(self, text: str) -> float:
        """Calculate financial-specific sentiment score"""
        words = re.findall(r'\b\w+\b', text.lower())

        positive_count = 0
        negative_count = 0
        total_words = len(words)

        if total_words == 0:
            return 0.0

        for word in words:
            if word in self.financial_lexicon['positive']:
                positive_count += 1
            elif word in self.financial_lexicon['negative']:
                negative_count += 1

        financial_score = (positive_count - negative_count) / total_words
        return max(-1, min(1, financial_score))

    def _ensemble_sentiment(self, sentiments: List[Dict]) -> Dict[str, float]:
        """Combine multiple sentiment analysis results"""
        weights = {'vader': 0.4, 'textblob': 0.3, 'financial': 0.3}

        weighted_scores = {
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0
        }

        total_weight = 0.0
        for sentiment in sentiments:
            method = sentiment.get('method', 'vader')
            weight = weights.get(method, 0.3)

            weighted_scores['compound'] += sentiment['compound'] * weight
            weighted_scores['positive'] += sentiment['positive'] * weight
            weighted_scores['negative'] += sentiment['negative'] * weight
            weighted_scores['neutral'] += sentiment['neutral'] * weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight

        return weighted_scores