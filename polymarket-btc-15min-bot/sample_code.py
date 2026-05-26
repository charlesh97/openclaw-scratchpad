"""
Reference implementation: Fusion Engine pattern from the Polymarket BTC 15-Minute Bot.

This shows the multi-signal fusion and self-learning weight optimization
pattern. Not a direct copy — a structural reference.
"""

from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class SignalProcessor:
    name: str
    compute: Callable[[dict], float]  # market data -> signal [-1, 1]
    weight: float = 1.0
    accuracy_history: list = field(default_factory=list)


class FusionEngine:
    """
    Weighted voting fusion engine with self-learning weight optimization.
    After each trade outcome, updates signal weights based on accuracy.
    """

    def __init__(self, signals: list[SignalProcessor]):
        self.signals = signals

    def evaluate(self, market_data: dict) -> tuple[float, dict]:
        """Returns fused signal [-1, 1] and per-signal breakdown."""
        votes = {}
        fused = 0.0
        total_weight = 0.0

        for sig in self.signals:
            vote = sig.compute(market_data)
            votes[sig.name] = {"signal": vote, "weight": sig.weight}
            fused += vote * sig.weight
            total_weight += sig.weight

        fused /= total_weight if total_weight > 0 else 1
        return fused, votes

    def update_weights(self, trade_outcome: float, lookback: int = 50):
        """
        After a trade resolves, update weights based on recent accuracy.
        trade_outcome: 1.0 for correct direction, -1.0 for wrong, 0.0 for neutral.
        """
        for sig in self.signals:
            sig.accuracy_history.append(trade_outcome)
            sig.accuracy_history = sig.accuracy_history[-lookback:]
            if len(sig.accuracy_history) >= 10:
                recent_accuracy = np.mean(
                    [abs(h) for h in sig.accuracy_history[-10:]]
                )
                sig.weight = 0.5 + 0.5 * recent_accuracy


# Signal implementations
def spike_detection(data: dict) -> float:
    """Detect sudden price spikes indicating momentum."""
    price = np.array(data.get("price_history", []))
    if len(price) < 5:
        return 0.0
    recent_volatility = np.std(price[-5:])
    baseline_volatility = np.std(price)
    if baseline_volatility == 0:
        return 0.0
    spike_ratio = recent_volatility / baseline_volatility
    return np.clip((spike_ratio - 1.5) / 3.0, -1.0, 1.0)


def sentiment_analysis(data: dict) -> float:
    """Basic news/social sentiment signal."""
    sentiment = data.get("sentiment_score", 0.0)
    return np.clip(sentiment, -1.0, 1.0)


def price_divergence(data: dict) -> float:
    """Detect divergence between exchanges."""
    coinbase = data.get("coinbase_price", 0)
    binance = data.get("binance_price", 0)
    if coinbase == 0 or binance == 0:
        return 0.0
    divergence = (coinbase - binance) / ((coinbase + binance) / 2)
    return np.clip(divergence * 5, -1.0, 1.0)
