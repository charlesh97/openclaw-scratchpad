"""
BTC 15-Minute Trading Bot — Core Signal Fusion Architecture (Reference Implementation)
Based on: https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class Signal:
    name: str
    value: float  # 0.0 to 1.0
    weight: float
    confidence: float


class SpikeDetector:
    """Detects price spikes across exchanges."""
    def detect(self, prices: List[float], window: int = 5) -> float:
        if len(prices) < window + 1:
            return 0.5
        recent = prices[-window:]
        mean = np.mean(recent)
        std = np.std(recent)
        last = prices[-1]
        if std == 0:
            return 0.5
        z_score = (last - mean) / std
        # Normalize to 0-1: z=0 → 0.5, z=2 → ~0.84, z=-2 → ~0.16
        return 1.0 / (1.0 + np.exp(-z_score))


class SentimentAnalyzer:
    """Analyzes news sentiment for BTC."""
    def analyze(self, news_headlines: List[str]) -> float:
        positive_keywords = ['bull', 'surge', 'upgrade', 'adoption', 'inflow']
        negative_keywords = ['ban', 'crash', 'hack', 'regulate', 'outflow']
        score = 0.5
        for headline in news_headlines:
            h = headline.lower()
            for kw in positive_keywords:
                if kw in h:
                    score += 0.05
            for kw in negative_keywords:
                if kw in h:
                    score -= 0.05
        return np.clip(score, 0.0, 1.0)


class DivergenceDetector:
    """Detects price divergence between exchanges."""
    def compute_divergence(self, exchange_prices: Dict[str, float]) -> float:
        prices = list(exchange_prices.values())
        if len(prices) < 2:
            return 0.5
        max_p = max(prices)
        min_p = min(prices)
        spread = (max_p - min_p) / np.mean(prices)
        # spread > 0.001 indicates significant divergence
        return np.clip(spread * 100, 0.0, 1.0)


class FusionEngine:
    """Weighted voting system with self-learning weight optimization."""
    
    def __init__(self):
        self.signals: List[Signal] = []
        self.learning_rate = 0.01
    
    def add_signal(self, signal: Signal):
        self.signals.append(signal)
    
    def compute_consensus(self) -> float:
        """Weighted average of all signals."""
        total_weight = sum(s.weight * s.confidence for s in self.signals)
        if total_weight == 0:
            return 0.5
        weighted_sum = sum(s.value * s.weight * s.confidence for s in self.signals)
        return weighted_sum / total_weight
    
    def update_weights(self, prediction: float, actual_outcome: float):
        """Simple gradient-based weight update."""
        error = prediction - actual_outcome
        for s in self.signals:
            gradient = error * s.value * s.confidence
            s.weight -= self.learning_rate * gradient
            s.weight = np.clip(s.weight, 0.0, 1.0)
    
    def should_trade(self, consensus: float, threshold_buy: float = 0.65,
                     threshold_sell: float = 0.35) -> str:
        if consensus >= threshold_buy:
            return "BUY_YES"
        elif consensus <= threshold_sell:
            return "BUY_NO"
        return "HOLD"
