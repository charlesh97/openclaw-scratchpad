"""
BTC 15-Minute Trading Bot - Signal Aggregation

Combines multiple signal sources for short-term BTC direction prediction.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Signal:
    name: str
    value: float  # 0 to 1 (probability of UP)
    weight: float
    
class SignalAggregator:
    """Combines multiple signal sources into a single prediction."""
    
    def __init__(self):
        self.signals: List[Signal] = []
        
    def add_technical_signal(self, ohlcv_data: np.ndarray) -> Signal:
        """RSI + MACD based signal."""
        closes = ohlcv_data[:, 3]
        rsi = self._compute_rsi(closes, 14)
        macd, signal = self._compute_macd(closes)
        
        # RSI < 30 oversold = buy signal, RSI > 70 overbought = sell signal
        rsi_signal = 1.0 - (rsi / 100.0) if rsi > 70 else (70 - rsi) / 70 if rsi < 30 else 0.5
        macd_signal = 0.5 + 0.5 * np.tanh((macd[-1] - signal[-1]) * 10)
        
        return Signal("technical", 0.5 * rsi_signal + 0.5 * macd_signal, 0.3)
    
    def add_orderbook_signal(self, order_book: dict) -> Signal:
        """Bid-ask imbalance signal."""
        total_bid_vol = sum(b[1] for b in order_book.get("bids", [])[:10])
        total_ask_vol = sum(a[1] for a in order_book.get("asks", [])[:10])
        
        if total_bid_vol + total_ask_vol == 0:
            return Signal("orderbook", 0.5, 0.2)
        
        imbalance = total_bid_vol / (total_bid_vol + total_ask_vol)
        return Signal("orderbook", imbalance, 0.2)
    
    def aggregate(self) -> float:
        """Weighted average of all signals."""
        if not self.signals:
            return 0.5
        
        total_weight = sum(s.weight for s in self.signals)
        weighted_sum = sum(s.value * s.weight for s in self.signals)
        return weighted_sum / total_weight
    
    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(prices)
        gains = deltas[deltas > 0].mean() if len(deltas[deltas > 0]) > 0 else 0
        losses = -deltas[deltas < 0].mean() if len(deltas[deltas < 0]) > 0 else 0.001
        rs = gains / losses
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, prices: np.ndarray):
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd = ema12 - ema26
        signal = self._ema(np.array([macd]), 9)
        return macd, signal[0] if len(signal) > 0 else 0
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        multiplier = 2 / (period + 1)
        result = data[0]
        for d in data[1:]:
            result = (d - result) * multiplier + result
        return result
