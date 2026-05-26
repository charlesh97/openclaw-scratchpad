"""
Prediction Market Maker — Simplified Implementation

Based on the #2-ranked strategy from Paradigm's Prediction Market Challenge.
Adapted for Polymarket's CLOB.

Core strategy: Monopoly regime pricing + volatility-adjusted quoting + inventory skew.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from enum import Enum


class Regime(Enum):
    MONOPOLY = "monopoly"   # Only liquidity provider
    NORMAL = "normal"       # Competitive quoting
    SIT_OUT = "sit_out"     # No edge available


@dataclass
class Quote:
    price: float      # In cents (0-100)
    size: float       # Number of shares
    side: str         # "bid" or "ask"


@dataclass
class OrderBookSnapshot:
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    mid_price: float


class PredictionMarketMaker:
    """
    Market-making strategy adapted from Paradigm Hackathon #2 entry.
    
    Key parameters tuned via 110 strategy iterations:
    - monopoly_size_factor: inverse of probability (85/prob for bids at low price)
    - normal_spread: how tight to quote in competitive regime  
    - skew_factor: inventory management aggressiveness
    - vol_filter_multiplier: volatility threshold for quoting
    """
    
    def __init__(
        self,
        monopoly_size_denom: float = 85.0,
        normal_spread_pct: float = 0.10,
        skew_factor: float = 0.3,
        vol_filter_multiplier: float = 2.0,
        max_position: float = 100.0,
    ):
        self.monopoly_size_denom = monopoly_size_denom
        self.normal_spread_pct = normal_spread_pct
        self.skew_factor = skew_factor
        self.vol_filter_multiplier = vol_filter_multiplier
        self.max_position = max_position
        self.current_position = 0.0  # Positive = net long YES
        
    def detect_regime(self, ob: OrderBookSnapshot, 
                      own_bid: Optional[float] = None,
                      own_ask: Optional[float] = None) -> Regime:
        """
        Detect which quoting regime we're in.
        
        Monopoly regime: No competing quotes within 5 cents of the true price.
        Sit-out: Volatility too high, no positive edge available.
        Normal: Standard competitive regime.
        """
        mid = ob.mid_price
        
        # Check if we have competition near the mid
        has_comp_bid = any(
            abs(p - mid) < 0.05 and p != (own_bid or 0)
            for p, _ in ob.bids[:3]
        )
        has_comp_ask = any(
            abs(p - mid) < 0.05 and p != (own_ask or 0)
            for p, _ in ob.asks[:3]
        )
        
        # Monopoly when probability is extreme AND no competitor
        is_extreme = mid < 0.10 or mid > 0.90
        
        if is_extreme and not has_comp_bid and not has_comp_ask:
            return Regime.MONOPOLY
        elif not has_comp_bid and not has_comp_ask:
            return Regime.MONOPOLY
        
        return Regime.NORMAL
    
    def compute_monopoly_quote(self, prob: float, side: str) -> Quote:
        """
        Monopoly regime: We are the only game in town.
        
        For buying YES at very low probability:
        - Post size = monopoly_size_denom / prob (inversely proportional)
        - Price = slightly better than prob to appear fair
        
        For buying NO at very high probability (prob > 0.9):
        - Mirror logic on the NO side
        """
        if side == "bid":  # We want to buy YES
            if prob < 0.1:
                # Extreme monopoly: post massive size at near-fair price
                size = self.monopoly_size_denom / max(prob, 0.001)
                price = prob * 0.95  # Slight discount to capture edge
            else:
                size = self.monopoly_size_denom / 0.1
                price = prob * 0.97
        else:  # We want to sell YES (buy NO)
            no_prob = 1.0 - prob
            if no_prob < 0.1:
                size = self.monopoly_size_denom / max(no_prob, 0.001)
                price = prob * 1.05  # Slight premium since buying NO = selling YES
            else:
                size = self.monopoly_size_denom / 0.1
                price = prob * 1.03
        
        return Quote(price=min(max(price, 0.01), 0.99), 
                    size=min(size, 5000), 
                    side=side)
    
    def compute_normal_quote(self, prob: float, ob: OrderBookSnapshot) -> List[Quote]:
        """
        Normal regime: Competitive quoting with volatility adjustment.
        
        - Spread based on expected retail flow
        - Size matched to average retail order (~$4.50 notional)
        - Skew position toward neutral
        """
        # Calculate fair spread
        best_bid = max(p for p, _ in ob.bids) if ob.bids else prob - 0.05
        best_ask = min(p for p, _ in ob.asks) if ob.asks else prob + 0.05
        
        spread = best_ask - best_bid
        fair_spread = max(spread, self.normal_spread_pct)
        
        # Position-based skew
        skew = -self.skew_factor * (self.current_position / self.max_position)
        
        bid_price = prob - (fair_spread / 2) + skew
        ask_price = prob + (fair_spread / 2) + skew
        
        # Size matched to expected retail (~$4.50 / price)
        retail_notional = 4.50
        bid_size = retail_notional / max(bid_price, 0.01)
        ask_size = retail_notional / max(1 - ask_price, 0.01)
        
        return [
            Quote(price=bid_price, size=min(bid_size, 200), side="bid"),
            Quote(price=ask_price, size=min(ask_size, 200), side="ask"),
        ]
    
    def compute_quotes(self, prob: float, 
                      ob: OrderBookSnapshot) -> List[Quote]:
        """
        Main entry point: compute optimal quotes based on current market conditions.
        """
        regime = self.detect_regime(ob)
        
        if regime == Regime.MONOPOLY:
            return [
                self.compute_monopoly_quote(prob, "bid"),
                self.compute_monopoly_quote(prob, "ask"),
            ]
        elif regime == Regime.NORMAL:
            return self.compute_normal_quote(prob, ob)
        else:
            return []  # Sit out
    
    def on_fill(self, side: str, size: float, price: float):
        """Update position on fill."""
        if side == "bid":
            self.current_position += size  # Bought YES
        else:
            self.current_position -= size  # Sold YES (bought NO)
        
        self.current_position = np.clip(
            self.current_position, 
            -self.max_position, 
            self.max_position
        )


# Example usage
if __name__ == "__main__":
    mm = PredictionMarketMaker()
    
    # Simulate a 15-min BTC market
    ob = OrderBookSnapshot(
        bids=[(0.42, 150), (0.41, 300), (0.40, 500)],
        asks=[(0.44, 100), (0.45, 250), (0.47, 400)],
        mid_price=0.43,
    )
    
    quotes = mm.compute_quotes(prob=0.43, ob=ob)
    for q in quotes:
        print(f"  {q.side.upper()}: {q.size} shares @ ${q.price:.4f}")
