#!/usr/bin/env python3
"""
Reference: Cross-Platform Arbitrage Bot
Based on realfishsam/prediction-market-arbitrage-bot

Simplified cross-platform arb detection logic.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketPrice:
    """Price quote from a platform."""
    platform: str  # 'polymarket' or 'kalshi'
    market_id: str
    best_bid: float
    best_ask: float
    timestamp: float

class CrossPlatformArbDetector:
    """
    Detects price discrepancies between Polymarket and Kalshi
    for the same event.
    """
    
    def __init__(self, min_edge: float = 0.01):
        self.min_edge = min_edge  # 1% minimum edge
    
    def check_arb(self,
                  poly: MarketPrice,
                  kalshi: MarketPrice) -> Optional[dict]:
        """
        Check if there's an arbitrage opportunity between
        the two platforms for the same event.
        
        Strategy: Buy on the cheaper platform, sell on the
        more expensive one.
        """
        # Find the spread
        if poly.best_bid > kalshi.best_ask:
            # Buy on Kalshi, sell on Polymarket
            edge = (poly.best_bid - kalshi.best_ask) / kalshi.best_ask
            if edge >= self.min_edge:
                return {
                    'buy_platform': 'kalshi',
                    'sell_platform': 'polymarket',
                    'buy_price': kalshi.best_ask,
                    'sell_price': poly.best_bid,
                    'edge_pct': edge * 100,
                    'action': 'BUY_KALSHI_SELL_POLY'
                }
        elif kalshi.best_bid > poly.best_ask:
            # Buy on Polymarket, sell on Kalshi
            edge = (kalshi.best_bid - poly.best_ask) / poly.best_ask
            if edge >= self.min_edge:
                return {
                    'buy_platform': 'polymarket',
                    'sell_platform': 'kalshi',
                    'buy_price': poly.best_ask,
                    'sell_price': kalshi.best_bid,
                    'edge_pct': edge * 100,
                    'action': 'BUY_POLY_SELL_KALSHI'
                }
        return None

if __name__ == "__main__":
    detector = CrossPlatformArbDetector(min_edge=0.02)
    print("Cross-Platform Arbitrage Detector (pmxt.dev)")
    print(f"Min edge: {detector.min_edge * 100:.0f}%")
