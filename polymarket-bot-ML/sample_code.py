"""
Polymarket ML Ensemble Bot — Feature Engineering & Bayesian Signal (Reference)
Based on: https://github.com/skharchikov/polymarket-bot
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MarketFeatures:
    """29 features extracted from market microstructure (simplified subset)."""
    # Price features
    mid_price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    
    # Volume features
    bid_volume_10: float  # cumulative volume within 10 ticks of best bid
    ask_volume_10: float
    volume_imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    # Time features
    seconds_to_resolution: float
    trades_last_minute: int
    volume_last_minute: float
    
    # Momentum features
    price_change_1m: float
    price_change_5m: float
    price_volatility_5m: float
    
    # Order book shape
    depth_ratio: float  # bid_depth / ask_depth at levels 1-5
    
    @classmethod
    def from_order_book(cls, order_book: dict) -> 'MarketFeatures':
        """Extract features from a Polymarket order book snapshot."""
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        best_bid = float(bids[0]['price']) if bids else 0.0
        best_ask = float(asks[0]['price']) if asks else 1.0
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = spread / mid * 10000 if mid > 0 else 0
        
        bid_vol_10 = sum(float(b['size']) for b in bids[:10])
        ask_vol_10 = sum(float(a['size']) for a in asks[:10])
        vol_imb = (bid_vol_10 - ask_vol_10) / (bid_vol_10 + ask_vol_10 + 1e-8)
        
        bid_depth = sum(float(b['size']) for b in bids[:5])
        ask_depth = sum(float(a['size']) for a in asks[:5])
        depth_ratio = bid_depth / (ask_depth + 1e-8)
        
        return cls(
            mid_price=mid, best_bid=best_bid, best_ask=best_ask,
            spread_bps=spread_bps, bid_volume_10=bid_vol_10,
            ask_volume_10=ask_vol_10, volume_imbalance=vol_imb,
            seconds_to_resolution=0.0, trades_last_minute=0,
            volume_last_minute=0.0, price_change_1m=0.0,
            price_change_5m=0.0, price_volatility_5m=0.0,
            depth_ratio=depth_ratio
        )


class BayesianSignalUpdater:
    """
    Bayesian updating of probability estimates.
    Combines prior belief with new signal evidence.
    """
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        # Beta distribution parameters
        self.alpha = prior_alpha
        self.beta = prior_beta
    
    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def update(self, signal_strength: float, signal_weight: float = 1.0):
        """
        Update posterior with a new signal.
        signal_strength: 0.0 (NO) to 1.0 (YES)
        signal_weight: confidence in this signal (0 to 1)
        """
        # Convert signal to pseudo-observations
        pseudo_yes = signal_strength * signal_weight * 10
        pseudo_no = (1 - signal_strength) * signal_weight * 10
        
        self.alpha += pseudo_yes
        self.beta += pseudo_no
    
    def credible_interval(self, alpha: float = 0.05) -> tuple:
        """95% credible interval for the probability estimate."""
        from scipy import stats
        lower = stats.beta.ppf(alpha / 2, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - alpha / 2, self.alpha, self.beta)
        return (lower, upper)


class KellyPositionSizer:
    """Kelly Criterion for optimal position sizing."""
    
    def __init__(self, fraction: float = 0.25):
        self.fraction = fraction  # Fraction of Kelly to use (conservative)
    
    def compute_size(self, probability: float, bankroll: float,
                     payout_if_yes: float = 1.0) -> float:
        """
        Compute optimal bet size using Kelly Criterion.
        probability: estimated probability of YES outcome
        """
        b = payout_if_yes  # net odds received on the bet
        p = probability
        q = 1 - p
        
        # Kelly fraction: (bp - q) / b
        kelly = (b * p - q) / b
        
        if kelly <= 0:
            return 0.0  # No edge, no bet
        
        return kelly * self.fraction * bankroll
