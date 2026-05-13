"""
Prediction Market Maker - Core Strategy (Adapted from Paradigm Challenge #2)

Key concepts: spread capture, inventory management, adverse selection avoidance.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Quote:
    bid: float
    ask: float
    bid_size: int
    ask_size: int

class PredictionMarketMaker:
    """Market maker for binary prediction markets."""
    
    def __init__(self, 
                 max_inventory: float = 100.0,
                 spread_bps: float = 10.0,  # 10 bps target spread
                 half_life_seconds: float = 300.0):
        self.max_inventory = max_inventory
        self.spread_bps = spread_bps
        self.half_life = half_life_seconds
        self.current_position = 0.0  # Net YES position
        self.true_probability = 0.5
        
    def estimate_true_prob(self, order_book: dict) -> float:
        """Estimate true probability from order book."""
        best_bid = order_book.get("bids", [])[0][0] if order_book.get("bids") else 0.0
        best_ask = order_book.get("asks", [])[0][0] if order_book.get("asks") else 1.0
        mid = (best_bid + best_ask) / 2
        
        # EWMA for smoothing
        self.true_probability = (
            0.3 * mid + 0.7 * self.true_probability
        )
        return self.true_probability
    
    def compute_quotes(self, order_book: dict) -> Optional[Quote]:
        """Compute bid/ask quotes with inventory risk adjustment."""
        prob = self.estimate_true_prob(order_book)
        
        # Base spread
        half_spread = self.spread_bps / 2 / 10000  # Convert to decimal
        
        # Inventory skew: if we're long YES, lower bid / raise ask
        inventory_skew = (self.current_position / self.max_inventory) * 0.02
        
        bid = prob - half_spread - inventory_skew
        ask = prob + half_spread - inventory_skew
        
        # Keep in [0, 1]
        bid = max(0.01, min(0.99, bid))
        ask = max(0.01, min(0.99, ask))
        
        # Size based on remaining inventory capacity
        bid_size = int((self.max_inventory - max(0, self.current_position)) / bid)
        ask_size = int((self.max_inventory - max(0, -self.current_position)) / (1 - ask))
        
        return Quote(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)
    
    def on_trade(self, side: str, price: float, size: int):
        """Update inventory after a trade fills."""
        if side == "sell":  # We sold YES (took our bid)
            self.current_position += size
        elif side == "buy":  # We bought YES (hit our ask) 
            self.current_position -= size
        
        # Mean-revert inventory if desired
        self.current_position *= 0.999  # Slow decay
