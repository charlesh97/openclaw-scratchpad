"""
Manifold Market Maker Bot — Python Adaptation Sketch
Original JS/TS: https://github.com/manifoldmarkets/market-maker
"""

import numpy as np


class EMAMarketMaker:
    """EMAV-based market maker for binary prediction markets."""

    def __init__(self, ema_alpha=0.1, emv_alpha=0.1, spread_mult=2.0):
        self.ema_alpha = ema_alpha
        self.emv_alpha = emv_alpha
        self.spread_mult = spread_mult
        self.ema = None
        self.emv = None

    def update(self, current_prob: float):
        """Update EMA and EMV with the latest market probability."""
        if self.ema is None:
            self.ema = current_prob
            self.emv = 0.0
        else:
            diff = current_prob - self.ema
            self.ema = self.ema_alpha * current_prob + (1 - self.ema_alpha) * self.ema
            self.emv = self.emv_alpha * abs(diff) + (1 - self.emv_alpha) * self.emv

    def get_orders(self):
        """Return bid/ask limit prices based on current EMA and volatility."""
        spread = self.spread_mult * self.emv
        bid = max(0.01, self.ema - spread / 2)
        ask = min(0.99, self.ema + spread / 2)
        return bid, ask
