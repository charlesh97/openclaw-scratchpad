"""
Unravelling the Probabilistic Forest - Arbitrage Detection

Reconstructed methodology from the paper for detecting
Market Rebalancing and Combinatorial Arbitrage.
"""

from dataclasses import dataclass
from typing import List, Tuple
import itertools

@dataclass
class MarketSnapshot:
    market_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    event_category: str

@dataclass
class ArbOpportunity:
    market_ids: List[str]
    type: str  # "rebalancing" or "combinatorial"
    guaranteed_profit: float
    max_size: int
    
class ProbabilisticForestArbDetector:
    def __init__(self):
        self.markets: List[MarketSnapshot] = []
    
    def detect_market_rebalancing(self) -> List[ArbOpportunity]:
        """Detect YES+NO < $1.00 within single markets."""
        opportunities = []
        for m in self.markets:
            # Cost to buy both YES and NO
            best_yes_price = m.yes_ask  # Must buy at ask
            best_no_price = m.no_ask    # Must buy at ask  
            total_cost = best_yes_price + best_no_price
            
            if total_cost < 0.98:  # At least 2% edge
                max_shares = min(
                    int(m.yes_ask * 1000),  # Limited by liquidity
                    int(m.no_ask * 1000)
                )
                opportunities.append(ArbOpportunity(
                    market_ids=[m.market_id],
                    type="rebalancing",
                    guaranteed_profit=(1.0 - total_cost) * max_shares,
                    max_size=max_shares
                ))
        return sorted(opportunities, key=lambda x: -x.guaranteed_profit)
    
    def detect_combinatorial(self) -> List[ArbOpportunity]:
        """
        Detect mispricing across related markets.
        E.g., "Will candidate A win?" and "Will candidate A win primary?"
        are logically linked - P(A wins general) <= P(A wins primary).
        """
        opportunities = []
        
        # Group markets by event category
        for category, cat_markets in self._group_by_category():
            # Check for price inconsistencies between related markets
            for m1, m2 in itertools.combinations(cat_markets, 2):
                if self._are_logically_linked(m1, m2):
                    # P(event A) must be >= P(event A AND event B)
                    p_m1 = (m1.yes_bid + m1.yes_ask) / 2
                    p_m2 = (m2.yes_bid + m2.yes_ask) / 2
                    
                    # If P(A) < P(A and B), we have an arb
                    if p_m1 < p_m2 * 0.95:  # 5% threshold
                        profit = (p_m2 - p_m1) * 100  # Per 100 shares
                        opportunities.append(ArbOpportunity(
                            market_ids=[m1.market_id, m2.market_id],
                            type="combinatorial",
                            guaranteed_profit=profit,
                            max_size=min(100, int(m1.yes_bid * 100))
                        ))
        
        return sorted(opportunities, key=lambda x: -x.guaranteed_profit)
    
    def _are_logically_linked(self, m1: MarketSnapshot, m2: MarketSnapshot) -> bool:
        """Check if two markets have logical dependency."""
        # Simplified - in practice would use event metadata
        return m1.event_category == m2.event_category
    
    def _group_by_category(self):
        """Group markets by event category for combinatorial analysis."""
        groups = {}
        for m in self.markets:
            groups.setdefault(m.event_category, []).append(m)
        return groups.items()
