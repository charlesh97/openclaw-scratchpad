"""
Reference implementation inspired by the 7-Phase BTC 15-min Trading Bot architecture.
This is a simplified educational version of the signal fusion and risk management pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    SPIKE = "spike"
    SENTIMENT = "sentiment"
    DIVERGENCE = "divergence"

@dataclass
class Signal:
    type: SignalType
    strength: float  # -1.0 to 1.0 (negative = bearish, positive = bullish)
    confidence: float  # 0.0 to 1.0
    timestamp: float

@dataclass
class FusionOutput:
    weighted_score: float
    confidence: float
    action: str  # "BUY_YES", "BUY_NO", "HOLD"

class SpikeDetector:
    """Detects rapid price movements in BTC price feeds."""
    
    def __init__(self, window: int = 10, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold
        self.history = []
    
    def update(self, price: float) -> Signal:
        self.history.append(price)
        if len(self.history) < self.window + 1:
            return Signal(SignalType.SPIKE, 0.0, 0.0, 0.0)
        
        prices = np.array(self.history[-self.window:])
        returns = np.diff(prices) / prices[:-1]
        z_score = (returns[-1] - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        strength = np.clip(z_score / self.threshold, -1.0, 1.0)
        confidence = min(abs(z_score) / self.threshold, 1.0)
        
        return Signal(SignalType.SPIKE, strength, confidence, time.time())

class SentimentAnalyzer:
    """Simple sentiment analyzer from news/social media signals."""
    
    def __init__(self):
        self.keywords = {
            "bullish": ["breakout", "rally", "support", "accumulation", "oversold"],
            "bearish": ["dump", "resistance", "distribution", "overbought", "correction"],
        }
    
    def analyze(self, text: str) -> Signal:
        text_lower = text.lower()
        bullish_count = sum(1 for k in self.keywords["bullish"] if k in text_lower)
        bearish_count = sum(1 for k in self.keywords["bearish"] if k in text_lower)
        
        net = (bullish_count - bearish_count) / max(bullish_count + bearish_count, 1)
        confidence = min((bullish_count + bearish_count) / 5.0, 1.0)
        
        return Signal(SignalType.SENTIMENT, net, confidence, time.time())

class DivergenceDetector:
    """Detects divergence between Polymarket probability and external reference price."""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
    
    def check_divergence(self, polymarket_prob: float, reference_prob: float) -> Signal:
        divergence = reference_prob - polymarket_prob
        strength = np.clip(divergence / self.threshold, -1.0, 1.0)
        confidence = min(abs(divergence) / self.threshold, 1.0)
        
        return Signal(SignalType.DIVERGENCE, strength, confidence, time.time())

class FusionEngine:
    """Weighted voting system that combines all signals."""
    
    def __init__(self):
        self.weights = {
            SignalType.SPIKE: 0.3,
            SignalType.SENTIMENT: 0.2,
            SignalType.DIVERGENCE: 0.5,
        }
        self.performance_history = []
    
    def fuse(self, signals: List[Signal]) -> FusionOutput:
        if not signals:
            return FusionOutput(0.0, 0.0, "HOLD")
        
        weighted_sum = sum(
            s.strength * self.weights[s.type] * s.confidence
            for s in signals
        )
        total_weight = sum(
            self.weights[s.type] * s.confidence
            for s in signals
        )
        
        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0.0
        score = weighted_sum / max(total_weight, 1e-8)
        score = np.clip(score, -1.0, 1.0)
        
        if score > 0.3:
            action = "BUY_YES"
        elif score < -0.3:
            action = "BUY_NO"
        else:
            action = "HOLD"
        
        return FusionOutput(score, avg_confidence, action)
    
    def update_weights(self, performance: Dict[SignalType, float]):
        """Self-learning: adjusts weights based on signal performance."""
        total = sum(performance.values())
        for signal_type in self.weights:
            if total > 0:
                self.weights[signal_type] = performance[signal_type] / total

class RiskManager:
    """Conservative risk management with dynamic sizing."""
    
    def __init__(self, max_per_trade: float = 1.0, max_drawdown: float = 0.3):
        self.max_per_trade = max_per_trade
        self.max_drawdown = max_drawdown
        self.peak_capital = 100.0
        self.current_capital = 100.0
    
    def calculate_position_size(self, signal: FusionOutput) -> float:
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if drawdown > self.max_drawdown:
            return 0.0  # Halt trading
        
        base_size = self.max_per_trade * abs(signal.weighted_score)
        confidence_scalar = signal.confidence
        
        if drawdown > 0.15:
            confidence_scalar *= 0.5  # Halve size during drawdown
        
        return min(base_size * confidence_scalar, self.max_per_trade)


# Usage example
if __name__ == "__main__":
    import time
    
    spike_detector = SpikeDetector()
    sentiment = SentimentAnalyzer()
    divergence = DivergenceDetector()
    fusion = FusionEngine()
    risk = RiskManager()
    
    # Simulate price data
    prices = [67000 + np.random.randn() * 500 for _ in range(20)]
    
    for price in prices:
        s1 = spike_detector.update(price)
        s2 = sentiment.analyze("BTC showing strong breakout above resistance levels")
        s3 = divergence.check_divergence(0.55, 0.62)  # Polymarket 55%, reference 62%
        
        fused = fusion.fuse([s1, s2, s3])
        size = risk.calculate_position_size(fused)
        
        print(f"Score: {fused.weighted_score:.3f}, Action: {fused.action}, Size: ${size:.2f}")
