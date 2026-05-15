"""
Reference implementation of the #2 Paradigm Challenge market-making strategy.
Core logic: monopoly regime detection + volatility-adjusted quoting + inventory skew.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Quote:
    price: int  # 1-99 cents
    quantity: int
    side: str  # "BID" or "ASK"

@dataclass
class OrderBookSnapshot:
    bids: list  # List of (price, quantity)
    asks: list  # List of (price, quantity)
    true_probability: float

class PredictionMarketMaker:
    """Market-making strategy inspired by the Paradigm Challenge #2 bot."""
    
    def __init__(self, base_spread: float = 0.08, position_limit: int = 100):
        self.base_spread = base_spread
        self.position_limit = position_limit
        self.inventory = 0  # Positive = long YES, Negative = long NO
        self.volatility_estimate = 0.0
        self.quote_history = []
    
    def estimate_volatility(self, recent_probabilities: list) -> float:
        """Estimate recent belief volatility."""
        if len(recent_probabilities) < 5:
            return 0.05
        returns = np.diff(recent_probabilities)
        return np.std(returns) * 2.0  # 2-sigma estimate
    
    def detect_monopoly_regime(self, snapshot: OrderBookSnapshot) -> Tuple[bool, bool]:
        """
        Detect if we would be the only liquidity provider at a price level.
        Returns (bid_monopoly, ask_monopoly).
        """
        if not snapshot.bids or not snapshot.asks:
            return False, False
        
        best_bid = snapshot.bids[0][0]
        best_ask = snapshot.asks[0][0]
        
        # Count unique makers at best levels
        bid_makers = len(set([b[1] for b in snapshot.bids if b[0] == best_bid]))
        ask_makers = len(set([a[1] for a in snapshot.asks if a[0] == best_ask]))
        
        return bid_makers <= 1, ask_makers <= 1
    
    def calculate_spread(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate optimal spread based on market conditions."""
        bid_monopoly, ask_monopoly = self.detect_monopoly_regime(snapshot)
        
        # Monopoly regime: quote wider spreads
        spread = self.base_spread
        if bid_monopoly:
            spread *= 1.5
        if ask_monopoly:
            spread *= 1.5
        
        # Volatility adjustment: widen during high volatility
        vol_adjustment = self.volatility_estimate * 2.0
        spread += vol_adjustment
        
        # Inventory adjustment
        inventory_skew = (self.inventory / self.position_limit) * 0.03
        spread -= inventory_skew  # Tighter when net flat
        
        return np.clip(spread, 0.02, 0.40)
    
    def calculate_inventory_skew(self) -> float:
        """
        Adjust mid-price based on inventory.
        If we're long YES, skew mid-price down (encourage selling).
        """
        return -self.inventory / self.position_limit * 0.02
    
    def generate_quotes(self, snapshot: OrderBookSnapshot,
                        recent_probabilities: list) -> Tuple[Optional[Quote], Optional[Quote]]:
        """Generate bid and ask quotes."""
        self.volatility_estimate = self.estimate_volatility(recent_probabilities)
        spread = self.calculate_spread(snapshot)
        skew = self.calculate_inventory_skew()
        
        mid_price = snapshot.true_probability + skew
        half_spread = spread / 2
        
        bid_price = max(1, int((mid_price - half_spread) * 100))
        ask_price = min(99, int((mid_price + half_spread) * 100))
        
        # Size based on expected retail flow (not fixed)
        base_size = 10
        bid_monopoly, ask_monopoly = self.detect_monopoly_regime(snapshot)
        
        bid_size = base_size * (2 if bid_monopoly else 1)
        ask_size = base_size * (2 if ask_monopoly else 1)
        
        # Reduce size if inventory is extreme
        inventory_ratio = abs(self.inventory) / self.position_limit
        if inventory_ratio > 0.5:
            bid_size = int(bid_size * (1 - inventory_ratio))
            ask_size = int(ask_size * (1 - inventory_ratio))
        
        bid = Quote(bid_price, bid_size, "BID")
        ask = Quote(ask_price, ask_size, "ASK")
        
        return bid, ask
    
    def on_fill(self, quote: Quote, fill_qty: int):
        """Update inventory on fill."""
        if quote.side == "BID":
            self.inventory += fill_qty
        else:
            self.inventory -= fill_qty

# Example usage
if __name__ == "__main__":
    mm = PredictionMarketMaker()
    
    # Simulated order book
    snapshot = OrderBookSnapshot(
        bids=[(45, 100), (44, 50)],
        asks=[(55, 80), (56, 40)],
        true_probability=0.50
    )
    
    recent_probs = [0.48, 0.51, 0.49, 0.52, 0.50, 0.47, 0.53]
    
    bid, ask = mm.generate_quotes(snapshot, recent_probs)
    print(f"BID: {bid.price}¢ x {bid.qty} | ASK: {ask.price}¢ x {ask.qty}")
    print(f"Spread: {ask.price - bid.price}¢ | Inventory: {mm.inventory}")
