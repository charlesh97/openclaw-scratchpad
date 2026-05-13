"""
Polybot - Strategy Reverse Engineering

Analyzes Polymarket trades to detect profitable strategies.
"""

from collections import defaultdict
import pandas as pd
from dataclasses import dataclass

@dataclass
class DetectedStrategy:
    name: str
    pattern: str
    avg_profit_per_trade: float
    total_trades: int
    win_rate: float

class StrategyDetector:
    """Detects profitable trading patterns from trade history."""
    
    def __init__(self):
        self.trades = []
    
    def add_trade(self, trade: dict):
        self.trades.append(trade)
    
    def detect_bundle_arb(self) -> DetectedStrategy:
        """Detect YES+NO < 1.0 bundle arbitrage trades."""
        bundle_trades = []
        for t in self.trades:
            if t.get("type") == "bundle_arb":
                bundle_trades.append(t)
        
        if not bundle_trades:
            return None
            
        df = pd.DataFrame(bundle_trades)
        return DetectedStrategy(
            name="Bundle Arbitrage",
            pattern="Buy both YES and NO when sum < 0.98",
            avg_profit_per_trade=df["profit"].mean(),
            total_trades=len(bundle_trades),
            win_rate=(df["profit"] > 0).mean()
        )
    
    def detect_cross_market_arb(self) -> DetectedStrategy:
        """Detect trades that profit from cross-market price discrepancies."""
        cross_trades = [t for t in self.trades if t.get("type") == "cross_market"]
        
        if not cross_trades:
            return None
        
        df = pd.DataFrame(cross_trades)
        return DetectedStrategy(
            name="Cross-Market Arbitrage",
            pattern="Trade related markets with pricing inconsistencies",
            avg_profit_per_trade=df["profit"].mean(),
            total_trades=len(cross_trades),
            win_rate=(df["profit"] > 0).mean()
        )
    
    def detect_market_making(self) -> DetectedStrategy:
        """Detect market making patterns (many small profitable trades)."""
        mm_trades = [t for t in self.trades if t.get("style") == "market_making"]
        
        if not mm_trades:
            return None
        
        df = pd.DataFrame(mm_trades)
        return DetectedStrategy(
            name="Market Making",
            pattern="Continuous bid-ask spread capture",
            avg_profit_per_trade=df["profit"].mean(),
            total_trades=len(mm_trades),
            win_rate=(df["profit"] > 0).mean()
        )
    
    def rank_strategies(self) -> list:
        """Rank detected strategies by Sharpe-like ratio."""
        strategies = filter(None, [
            self.detect_bundle_arb(),
            self.detect_cross_market_arb(),
            self.detect_market_making()
        ])
        return sorted(strategies, key=lambda s: s.avg_profit_per_trade * s.win_rate, reverse=True)
