# news_analyzer/event_extractor_compatible.py
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


class CompatibleEventExtractor:
    def __init__(self):
        self.event_patterns = self._initialize_event_patterns()

    def _initialize_event_patterns(self) -> Dict:
        """Initialize patterns for detecting financial events"""
        return {
            'earnings': {
                'patterns': [
                    r'earnings', r'quarterly', r'Q[1-4]', r'profit', r'revenue',
                    r'EPS', r'beat estimates', r'missed estimates', r'results'
                ],
                'weight': 0.9
            },
            'mergers_acquisitions': {
                'patterns': [
                    r'acquire', r'merger', r'buyout', r'takeover', r'deal',
                    r'acquisition', r'merge with', r'purchase'
                ],
                'weight': 0.8
            },
            'product_news': {
                'patterns': [
                    r'launch', r'new product', r'unveil', r'release', r'announce',
                    r'introduce', r'innovative', r'breakthrough'
                ],
                'weight': 0.6
            },
            'regulatory': {
                'patterns': [
                    r'FDA', r'approval', r'regulatory', r'commission', r'authority',
                    r'clearance', r'approve', r'rejection'
                ],
                'weight': 0.7
            },
            'leadership': {
                'patterns': [
                    r'CEO', r'executive', r'appoint', r'resign', r'leadership',
                    r'management', r'hire', r'departure'
                ],
                'weight': 0.5
            },
            'economic': {
                'patterns': [
                    r'inflation', r'interest rates', r'GDP', r'employment',
                    r'economy', r'recession', r'growth', r'market'
                ],
                'weight': 0.4
            }
        }

    def extract_events_from_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Extract structured events from news data"""
        events = []

        for _, row in news_df.iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
            detected_events = self._analyze_text_for_events(text, row)

            for event_type, confidence in detected_events.items():
                impact_score = self._calculate_impact_score(event_type, confidence, row)

                events.append({
                    'date': row['date'],
                    'ticker': row.get('ticker', 'Unknown'),
                    'event_type': event_type,
                    'confidence': confidence,
                    'impact_score': impact_score,
                    'sentiment': row.get('sentiment_score', 0),
                    'title': row.get('title', ''),
                    'source': row.get('source', '')
                })

        return pd.DataFrame(events)

    def _analyze_text_for_events(self, text: str, news_item: Dict) -> Dict[str, float]:
        """Analyze text for financial events"""
        events_detected = {}

        for event_type, event_info in self.event_patterns.items():
            confidence = self._calculate_event_confidence(text, event_info['patterns'])
            if confidence > 0.3:  # Minimum confidence threshold
                # Adjust confidence based on event weight
                adjusted_confidence = confidence * event_info['weight']
                events_detected[event_type] = adjusted_confidence

        return events_detected

    def _calculate_event_confidence(self, text: str, patterns: List[str]) -> float:
        """Calculate confidence score for event detection"""
        if not text:
            return 0.0

        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1

        return matches / len(patterns)

    def _calculate_impact_score(self, event_type: str, confidence: float, news_item: Dict) -> float:
        """Calculate impact score for detected event"""
        base_impacts = {
            'earnings': 0.8,
            'mergers_acquisitions': 0.7,
            'regulatory': 0.9,
            'product_news': 0.5,
            'leadership': 0.4,
            'economic': 0.6
        }

        base_impact = base_impacts.get(event_type, 0.5)
        sentiment_effect = abs(news_item.get('sentiment_score', 0)) * 0.3

        impact_score = base_impact * confidence * (1 + sentiment_effect)
        return min(impact_score, 1.0)