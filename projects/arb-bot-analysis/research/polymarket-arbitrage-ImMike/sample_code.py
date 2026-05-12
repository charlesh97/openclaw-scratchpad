"""
Polymarket Arbitrage Bot — Core arbitrage detection logic.

From ImMike's polymarket-arbitrage repo.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ArbitrageOpportunity:
    market_id: str
    market_name: str
    type: str  # "bundle", "cross_platform", "market_making"
    profit_bps: float       # profit in basis points
    total_cost: float       # combined cost per share
    max_shares: int         # max executable size from order book depth
    confidence: float       # 0.0 - 1.0


class BundleArbitrageDetector:
    """
    Detects YES + NO bundle pricing inefficiencies.
    If YES + NO < $1.00, buying both guarantees risk-free profit at resolution.
    """
    
    def __init__(self, min_profit_bps: float = 1.0):
        self.min_profit_bps = min_profit_bps
    
    def check_opportunity(self, market_id: str, yes_ask: float, 
                          no_ask: float, yes_depth: int, no_depth: int,
                          market_name: str = "") -> Optional[ArbitrageOpportunity]:
        total_cost = yes_ask + no_ask
        
        if total_cost >= 1.0:
            return None
        
        profit_bps = (1.0 - total_cost) * 10000  # convert to bps
        
        if profit_bps < self.min_profit_bps:
            return None
        
        max_shares = min(yes_depth, no_depth)
        
        return ArbitrageOpportunity(
            market_id=market_id,
            market_name=market_name or market_id,
            type="bundle",
            profit_bps=profit_bps,
            total_cost=total_cost,
            max_shares=max_shares,
            confidence=0.95 if profit_bps > 50 else 0.85,
        )


class CrossPlatformArbitrageDetector:
    """
    Detects price differences between Polymarket and Kalshi
    for the same prediction market (e.g., "BTC > $100k in 1 hour").
    """
    
    def check_opportunity(self, polymarket_price: float, kalshi_price: float,
                          polymarket_market_id: str, kalshi_ticker: str) -> Optional[dict]:
        spread = abs(polymarket_price - kalshi_price)
        
        if spread < 0.01:  # less than 1 cent — no opportunity
            return None
        
        # Buy the cheaper platform's YES, sell (or buy NO on) the expensive platform
        if polymarket_price < kalshi_price:
            entry_market = "Polymarket"
            hedge_market = "Kalshi"
        else:
            entry_market = "Kalshi"
            hedge_market = "Polymarket"
        
        return {
            "spread": spread,
            "entry_market": entry_market,
            "hedge_market": hedge_market,
            "polymarket_price": polymarket_price,
            "kalshi_price": kalshi_price,
            "profit_potential_pct": spread * 100,
        }


# Configuration example
CONFIG = {
    "mode": {
        "data_mode": "simulation",  # or "real"
        "trading_mode": "paper",    # or "live"
    },
    "risk_management": {
        "max_position_size_usdc": 500,
        "daily_loss_limit_usdc": 200,
        "kill_switch_enabled": True,
    },
    "arbitrage": {
        "min_profit_bps": 5,
        "scan_interval_ms": 1000,
    },
    "dashboard": {
        "enabled": True,
        "port": 8080,
    },
}
