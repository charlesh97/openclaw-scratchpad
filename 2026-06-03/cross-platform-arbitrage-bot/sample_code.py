"""
Cross-Platform Arbitrage Bot — Core Detection Logic
Source: https://github.com/ImMike/polymarket-arbitrage
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketPair:
    """A single prediction market outcome."""
    platform: str  # "Polymarket" or "Kalshi"
    market_id: str
    yes_price: float   # cents, e.g. 0.45
    no_price: float    # cents, e.g. 0.52
    liquidity: float   # USDC available at those prices


@dataclass
class ArbitrageOpportunity:
    arb_type: str  # "bundle", "cross_platform"
    buy_side: str  # "YES" or "NO"
    buy_price: float
    sell_side: str
    sell_price: float
    expected_profit_pct: float
    max_size: float  # dollar amount fillable


def detect_bundle_arb(market: MarketPair, min_edge: float = 0.01) -> Optional[ArbitrageOpportunity]:
    """Bundle arb: YES + NO < $1.00 → buy both, collect the spread at settlement."""
    total = market.yes_price + market.no_price
    profit_pct = 1.0 - total
    if profit_pct >= min_edge:
        size = min(market.liquidity, 100.0)  # conservative sizing
        return ArbitrageOpportunity(
            arb_type="bundle",
            buy_side="YES+NO",
            buy_price=total,
            sell_side="SETTLEMENT",
            sell_price=1.0,
            expected_profit_pct=profit_pct * 100,
            max_size=size,
        )
    return None


def detect_cross_platform_arb(
    polymarket_market: MarketPair,
    kalshi_market: MarketPair,
    min_edge: float = 0.01,
) -> Optional[ArbitrageOpportunity]:
    """Cross-platform: buy cheaper YES on one platform, sell on the other."""
    # Buy YES on cheaper platform, sell on the other
    if kalshi_market.yes_price < polymarket_market.yes_price - min_edge:
        return ArbitrageOpportunity(
            arb_type="cross_platform",
            buy_side="YES",
            buy_price=kalshi_market.yes_price,
            sell_side="YES",
            sell_price=polymarket_market.yes_price,
            expected_profit_pct=(
                polymarket_market.yes_price - kalshi_market.yes_price
            ) * 100,
            max_size=min(kalshi_market.liquidity, polymarket_market.liquidity),
        )
    elif polymarket_market.yes_price < kalshi_market.yes_price - min_edge:
        return ArbitrageOpportunity(
            arb_type="cross_platform",
            buy_side="YES",
            buy_price=polymarket_market.yes_price,
            sell_side="YES",
            sell_price=kalshi_market.yes_price,
            expected_profit_pct=(
                kalshi_market.yes_price - polymarket_market.yes_price
            ) * 100,
            max_size=min(polymarket_market.liquidity, kalshi_market.liquidity),
        )
    return None
