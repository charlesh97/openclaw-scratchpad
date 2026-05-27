"""
Paradigm Market Maker — Core Strategy (Python adaptation)

Based on the #2 ranked entry from Paradigm's Prediction Market Challenge.
Adapted from octavi42/prediction-market-maker for Polymarket CLOB.

Key concepts:
- Quote passive limit orders around estimated fair value
- Adjust spreads based on volatility (wider when volatile)
- Manage inventory via price skew
- Exploit "monopoly regime" when probability near extremes
"""

import math
import numpy as np


class ParadigmMarketMaker:
    """
    Market-making strategy for binary prediction markets.
    
    Parameters:
        kelly_fraction: Fraction of Kelly Criterion to use (0-1)
        base_spread: Base bid-ask spread in probability points
        vol_adjustment: How aggressively to widen spread in high vol
        skew_factor: How aggressively to skew quotes for inventory mgmt
        monopoly_threshold: Probability threshold for monopoly regime detection
    """
    
    def __init__(
        self,
        kelly_fraction=0.25,
        base_spread=0.04,
        vol_adjustment=2.0,
        skew_factor=0.3,
        monopoly_threshold=0.15,
        max_position=100,
    ):
        self.kelly_fraction = kelly_fraction
        self.base_spread = base_spread
        self.vol_adjustment = vol_adjustment
        self.skew_factor = skew_factor
        self.monopoly_threshold = monopoly_threshold
        self.max_position = max_position
        self.current_position = 0  # positive = long YES, negative = long NO
        self.capital = 1000.0
    
    def estimate_true_probability(self, bid, ask, external_signal=None):
        """
        Estimate the true probability from order book + external signals.
        
        In the Paradigm simulation, true probability was known.
        In production, estimate from CLOB mid-price and external data.
        """
        mid = (bid + ask) / 2
        if external_signal is not None:
            # Blend mid-price with external signal (e.g., ML model, news sentiment)
            return 0.7 * mid + 0.3 * external_signal
        return mid
    
    def compute_volatility(self, price_history):
        """
        Compute belief volatility from recent price history.
        Returns a volatility estimate in [0, 1].
        """
        if len(price_history) < 2:
            return 0.02
        returns = np.diff(np.log(np.clip(price_history, 0.01, 0.99)))
        return np.std(returns) * math.sqrt(252)  # Annualized
    
    def compute_spread(self, estimated_prob, volatility):
        """
        Dynamic spread: wider when volatile, tighter when confident.
        """
        vol_component = self.vol_adjustment * volatility
        # Tighter spreads near extremes (monopoly regime potential)
        prob_extreme = min(estimated_prob, 1 - estimated_prob)
        extreme_bonus = max(0, (self.monopoly_threshold - prob_extreme) / self.monopoly_threshold) * 0.5
        
        spread = self.base_spread * (1 + vol_component) * (1 - 0.3 * extreme_bonus)
        return min(spread, 0.15)  # Cap spread at 15 cents
    
    def detect_monopoly_regime(self, estimated_prob, order_book_depth):
        """
        Detect when the market is in a "monopoly regime" where one side
        has little to no competitive quoting.
        
        In the original: ~60% of edge came from this regime.
        """
        is_extreme = estimated_prob < self.monopoly_threshold or estimated_prob > (1 - self.monopoly_threshold)
        
        if not is_extreme:
            return False, 0
        
        # Check if one side of the book is thin
        if estimated_prob < 0.5:
            side_imbalance = order_book_depth.get('ask_depth', 0) / max(order_book_depth.get('bid_depth', 1), 1)
        else:
            side_imbalance = order_book_depth.get('bid_depth', 0) / max(order_book_depth.get('ask_depth', 1), 1)
        
        is_monopoly = side_imbalance > 3.0  # One side has 3x the depth
        monopoly_edge = min(side_imbalance / 10.0, 0.05)  # Extra edge up to 5 cents
        
        return is_monopoly, monopoly_edge
    
    def compute_kelly_size(self, estimated_prob, price):
        """
        Kelly Criterion position sizing.
        
        For a binary bet at price p with estimated true probability q:
        f* = (q * (1-p) - (1-q) * p) / (p * (1-p))
        """
        if price <= 0 or price >= 1:
            return 0
        
        expected_value = estimated_prob * (1 - price) - (1 - estimated_prob) * price
        if expected_value <= 0:
            return 0
        
        # Full Kelly for binary outcomes
        edge = 2 * estimated_prob - 1  # simplified
        kelly = edge / (2 * price - 1) if abs(2 * price - 1) > 0.001 else 0
        
        # Fractional Kelly
        return max(0, kelly * self.kelly_fraction)
    
    def get_quotes(self, bid, ask, order_book_depth, price_history, external_signal=None):
        """
        Main entry point: compute bid and ask quotes.
        
        Returns:
            dict with 'bid_price', 'bid_size', 'ask_price', 'ask_size'
        """
        estimated_prob = self.estimate_true_probability(bid, ask, external_signal)
        volatility = self.compute_volatility(price_history)
        spread = self.compute_spread(estimated_prob, volatility)
        
        # Inventory skew: shift quotes to reduce position
        inventory_ratio = self.current_position / max(self.max_position, 1)
        inventory_skew = self.skew_factor * inventory_ratio * spread
        
        # Monopoly regime detection
        is_monopoly, monopoly_edge = self.detect_monopoly_regime(estimated_prob, order_book_depth)
        
        # Calculate quote prices
        half_spread = spread / 2
        my_bid = estimated_prob - half_spread + inventory_skew
        my_ask = estimated_prob + half_spread + inventory_skew
        
        if is_monopoly:
            # Lean harder into the thin side
            if estimated_prob < 0.5:
                my_bid -= monopoly_edge  # Offer better bid to capture monopoly
            else:
                my_ask += monopoly_edge
        
        # Clamp to [0.01, 0.99]
        my_bid = max(0.01, min(my_bid, 0.99))
        my_ask = max(0.01, min(my_ask, 0.99))
        
        # Position sizing
        bid_size = self.compute_kelly_size(estimated_prob, my_bid)
        ask_size = self.compute_kelly_size(1 - estimated_prob, 1 - my_ask)
        
        return {
            'bid_price': my_bid,
            'bid_size': bid_size * self.capital,
            'ask_price': my_ask,
            'ask_size': ask_size * self.capital,
        }
    
    def on_fill(self, side, price, quantity):
        """Update position after a fill."""
        if side == 'buy':
            self.current_position += quantity
        else:
            self.current_position -= quantity
        self.capital -= price * quantity if side == 'buy' else -price * quantity
