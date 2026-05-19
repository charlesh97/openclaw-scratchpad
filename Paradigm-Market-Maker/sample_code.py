"""
Reference implementation based on octavi42's Paradigm market maker strategy.

Core concept: regime-aware market making with volatility-adjusted quoting.
"""

import numpy as np
from typing import Tuple, Optional

class RegimeAwareMarketMaker:
    """
    A market maker that adapts between monopoly and normal quoting regimes.
    
    Key insight from Paradigm #2 strategy:
    - ~60% edge comes from monopoly regime (wide spread, high volume when competition is thin)
    - ~40% edge comes from normal regime (tight spread, volatility-adjusted)
    """
    
    def __init__(self, 
                 base_spread: float = 0.04,
                 max_position: float = 100.0,
                 inventory_skew_factor: float = 0.3,
                 volatility_threshold: float = 0.02):
        self.base_spread = base_spread
        self.max_position = max_position
        self.inventory_skew_factor = inventory_skew_factor
        self.volatility_threshold = volatility_threshold
        self.inventory = 0.0
        
    def estimate_true_prob(self, bid: float, ask: float) -> float:
        """Estimate true probability from order book mid."""
        return (bid + ask) / 2.0
    
    def compute_volatility(self, prices: np.ndarray) -> float:
        """Compute recent price volatility."""
        if len(prices) < 2:
            return 0.01
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def detect_monopoly_regime(self, book_depth_bid: float, 
                                 book_depth_ask: float) -> bool:
        """
        Detect if we're in monopoly regime (thin competition on one side).
        
        In the hackathon, monopoly = competitor ladder has been mostly consumed
        and not yet replenished, leaving us as the primary liquidity provider.
        """
        thin_threshold = 500  # shares total on both sides
        return (book_depth_bid + book_depth_ask) < thin_threshold
    
    def compute_quotes(self, 
                       fair_prob: float,
                       volatility: float,
                       book_depth_bid: float,
                       book_depth_ask: float) -> Tuple[float, float, float, float]:
        """
        Compute bid/ask prices and sizes.
        
        Returns: (bid_price, bid_size, ask_price, ask_size)
        """
        is_monopoly = self.detect_monopoly_regime(book_depth_bid, book_depth_ask)
        
        # Volatility adjustment: widen spread in high volatility
        vol_multiplier = 1.0 + (volatility / self.volatility_threshold) * 2.0
        
        if is_monopoly:
            # Monopoly regime: wider spread, larger size
            spread = self.base_spread * 3.0 * vol_multiplier
            size_mult = 4.0
        else:
            # Normal regime: tight spread, normal size
            spread = self.base_spread * vol_multiplier
            size_mult = 1.0
        
        # Inventory skew: move quotes to reduce inventory
        inventory_skew = (self.inventory / self.max_position) * self.inventory_skew_factor
        
        bid_price = fair_prob - spread / 2 - inventory_skew
        ask_price = fair_prob + spread / 2 - inventory_skew
        
        # Size proportional to inverse of volatility
        base_size = 100 * size_mult
        size = base_size / max(volatility * 10, 1.0)
        
        return (bid_price, size, ask_price, size)
    
    def update_inventory(self, trade_price: float, 
                          fair_prob: float, quantity: float, 
                          is_buy: bool) -> None:
        """Update inventory after a fill."""
        if is_buy:
            self.inventory += quantity
        else:
            self.inventory -= quantity


# Example usage
if __name__ == "__main__":
    mm = RegimeAwareMarketMaker()
    
    # Simulate a step
    prices = np.array([0.52, 0.53, 0.51, 0.54, 0.52])
    vol = mm.compute_volatility(prices)
    
    bid_p, bid_s, ask_p, ask_s = mm.compute_quotes(
        fair_prob=0.53,
        volatility=vol,
        book_depth_bid=200,
        book_depth_ask=150
    )
    
    print(f"Fair prob: 0.53")
    print(f"Volatility: {vol:.4f}")
    print(f"Bid: {bid_p:.3f} x {bid_s:.0f}")
    print(f"Ask: {ask_p:.3f} x {ask_s:.0f}")
    print(f"Spread: {ask_p - bid_p:.4f}")
