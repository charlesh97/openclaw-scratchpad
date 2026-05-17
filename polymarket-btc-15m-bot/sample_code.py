"""
Reference architecture for the 7-phase Polymarket BTC 15-min trading bot.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from enum import Enum

@dataclass
class Signal:
    source: str
    direction: float  # +1 (up), -1 (down), 0 (neutral)
    confidence: float  # 0.0 to 1.0
    weight: float      # dynamic, updated by learning

class SignalType(Enum):
    SPIKE = "spike"
    SENTIMENT = "sentiment"
    DIVERGENCE = "divergence"

class SpikeDetector:
    """Detects rapid price movements across exchanges."""
    
    def detect(self, prices: Dict[str, np.ndarray]) -> Signal:
        cross_exchange_spread = max(prices.keys()) - min(prices.keys())
        if cross_exchange_spread > 0.005:  # 50 bps threshold
            return Signal("spike", 
                        direction=1.0 if np.mean(list(prices.values())[-1]) > np.mean(list(prices.values())[0]) else -1.0,
                        confidence=min(cross_exchange_spread * 100, 0.9),
                        weight=0.3)
        return Signal("spike", 0, 0, 0.3)

class SentimentAnalyzer:
    """Analyzes crypto news and social sentiment."""
    
    def analyze(self, news_headlines: List[str]) -> Signal:
        positive_keywords = ['bullish', 'surge', 'breakout', 'etf', 'institutional']
        negative_keywords = ['crash', 'ban', 'hack', 'regulation', 'sell-off']
        
        pos_count = sum(1 for h in news_headlines for k in positive_keywords if k in h.lower())
        neg_count = sum(1 for h in news_headlines for k in negative_keywords if k in h.lower())
        
        net = pos_count - neg_count
        total = pos_count + neg_count or 1
        
        return Signal("sentiment",
                     direction=np.sign(net),
                     confidence=abs(net) / total,
                     weight=0.25)

class DivergenceDetector:
    """Detects divergence between Polymarket prices and external markets."""
    
    def detect(self, polymarket_prob: float, external_prob: float) -> Signal:
        divergence = polymarket_prob - external_prob
        threshold = 0.03
        
        if abs(divergence) > threshold:
            # Market should mean-revert
            direction = -np.sign(divergence)
            confidence = min(abs(divergence) * 5, 0.95)
            return Signal("divergence", direction, confidence, 0.35)
        return Signal("divergence", 0, 0, 0.35)

class FusionEngine:
    """Weighted voting system that fuses signals."""
    
    def __init__(self):
        self.signals: List[Signal] = []
        
    def fuse(self, signals: List[Signal]) -> float:
        """
        Weighted voting: sum(signal.direction * signal.confidence * signal.weight)
        / sum(signal.weight * signal.confidence) 
        
        Returns: -1.0 to +1.0 (negative = down, positive = up)
        """
        weighted_sum = sum(s.direction * s.confidence * s.weight for s in signals)
        weight_sum = sum(s.confidence * s.weight for s in signals)
        
        if weight_sum == 0:
            return 0.0
        return weighted_sum / weight_sum

class SelfLearningOptimizer:
    """Optimizes signal weights based on historical performance."""
    
    def __init__(self, learning_rate=0.01, window=100):
        self.learning_rate = learning_rate
        self.window = window
        self.history: List[Dict] = []
    
    def update_weights(self, signals: List[Signal], outcome: float):
        """Update weights using gradient-based optimization."""
        self.history.append({"signals": signals, "outcome": outcome})
        
        if len(self.history) < self.window:
            return
        
        # Recent performance tracking
        recent = self.history[-self.window:]
        for signal in signals:
            matches = sum(1 for h in recent 
                         if s := next((s for s in h["signals"] 
                                      if s.source == signal.source), None)
                         if s.direction * h["outcome"] > 0)
            accuracy = matches / self.window
            
            # Increase weight for accurate signals, decrease for inaccurate
            signal.weight *= (1 + self.learning_rate * (accuracy - 0.5))
            signal.weight = max(0.05, min(0.5, signal.weight))


# Complete pipeline
class TradingBot:
    def __init__(self):
        self.spike_detector = SpikeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.divergence_detector = DivergenceDetector()
        self.fusion_engine = FusionEngine()
        self.optimizer = SelfLearningOptimizer()
        self.max_risk_per_trade = 1.0  # $1 max per trade
    
    def run_step(self, market_data) -> Optional[dict]:
        # Phase 1-4: Signal processing
        spike = self.spike_detector.detect(market_data['exchange_prices'])
        sentiment = self.sentiment_analyzer.analyze(market_data['news_headlines'])
        divergence = self.divergence_detector.detect(
            market_data['polymarket_prob'],
            market_data['external_prob']
        )
        
        # Phase 5: Fusion
        fused_signal = self.fusion_engine.fuse([spike, sentiment, divergence])
        
        if abs(fused_signal) < 0.3:  # Minimum conviction threshold
            return None
        
        # Phase 6: Risk management
        position_size = self.max_risk_per_trade * abs(fused_signal)
        
        return {
            'direction': 'up' if fused_signal > 0 else 'down',
            'size': position_size,
            'confidence': abs(fused_signal)
        }
