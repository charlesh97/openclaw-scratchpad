"""
Prediction Market Maker — Paradigm Challenge Strategy (Reference)
Based on: https://github.com/octavi42/prediction-market-maker
Key insight: ~60% of edge comes from monopoly regime
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    price: int  # 1-99 cents (integer ticks)
    size: float
    side: str  # 'bid' or 'ask'


@dataclass
class BookState:
    best_bid: int
    best_ask: int
    bid_sizes: dict  # price -> size
    ask_sizes: dict
    mid_price: float
    true_probability: float  # known only to arbitrageur in simulation


class VolatilityAdjustedQuoter:
    """Adjusts quoting aggressiveness based on recent volatility."""
    
    def __init__(self, base_spread: int = 4):  # base 4-cent spread
        self.base_spread = base_spread
        self.recent_prices = []
    
    def add_price(self, price: float):
        self.recent_prices.append(price)
        if len(self.recent_prices) > 20:
            self.recent_prices.pop(0)
    
    def compute_volatility(self) -> float:
        if len(self.recent_prices) < 5:
            return 0.0
        return float(np.std(self.recent_prices[-5:]))
    
    def adjust_spread(self) -> int:
        """Widen spread in high volatility, narrow in low."""
        vol = self.compute_volatility()
        adjustment = int(vol * 20)  # scale vol to ticks
        return min(self.base_spread + adjustment, 20)


class MonopolyDetector:
    """Detects when we're the only liquidity on one side."""
    
    def is_monopoly(self, book: BookState, our_orders: list,
                    side: str) -> bool:
        """Check if we're the only liquidity provider on a side."""
        if side == 'bid':
            other_bids = [o for o in our_orders if o.side != 'bid']
            return len(other_bids) == 0
        else:
            other_asks = [o for o in our_orders if o.side != 'ask']
            return len(other_asks) == 0
    
    def monopoly_edge_multiplier(self, is_monopoly: bool) -> float:
        """In monopoly regime, we can quote wider."""
        return 1.5 if is_monopoly else 1.0


class InventoryManager:
    """Manages inventory skew to prevent catastrophic losses."""
    
    def __init__(self, max_inventory: float = 100.0, skew_scale: float = 0.5):
        self.current_inventory = 0.0  # positive = net long YES
        self.max_inventory = max_inventory
        self.skew_scale = skew_scale
    
    def compute_inventory_skew(self) -> float:
        """
        Returns a skew adjustment: positive = bias toward selling,
        negative = bias toward buying.
        """
        inventory_pct = self.current_inventory / self.max_inventory
        return -inventory_pct * self.skew_scale
    
    def update_inventory(self, filled_qty: float, side: str):
        if side == 'bid':
            self.current_inventory += filled_qty  # bought YES
        else:
            self.current_inventory -= filled_qty  # sold YES


class OrderSizer:
    """
    Sizes orders to match expected retail order flow.
    Key insight from Paradigm challenge: sizing matters more than params.
    """
    
    def __init__(self, expected_retail_per_step: float = 4.5,
                 num_levels: int = 3):
        self.expected_retail = expected_retail_per_step
        self.num_levels = num_levels
    
    def compute_size_at_level(self, level: int, is_monopoly: bool = False) -> float:
        """
        Distribute expected retail flow across price levels.
        """
        if is_monopoly:
            base = self.expected_retail * 2  # capture more in monopoly
        else:
            base = self.expected_retail
        
        # More at top level, less at deeper levels
        weights = [0.5, 0.3, 0.2][:self.num_levels]
        return base * weights[level]
