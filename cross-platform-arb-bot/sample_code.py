"""
Cross-Platform Arbitrage Bot - Core Detection Logic

Detects price discrepancies between Polymarket and Kalshi.
Uses pmxt.dev as the unified API bridge.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

@dataclass
class ArbOpportunity:
    event_id: str
    polymarket_price: float
    kalshi_price: float
    direction: str  # "buy_polymarket_sell_kalshi" or vice versa
    profit_per_contract: float
    max_contracts: int
    
class CrossPlatformArbBot:
    def __init__(self, min_profit_pct: float = 0.5, max_position: int = 1000):
        self.min_profit_pct = min_profit_pct
        self.max_position = max_position
        
    async def find_opportunities(self) -> list[ArbOpportunity]:
        """Scan both platforms for matching events with price gaps."""
        # Fetch all active markets from both platforms
        polymarket_markets = await self._fetch_polymarket_markets()
        kalshi_markets = await self._fetch_kalshi_markets()
        
        opportunities = []
        matched = self._match_identical_events(polymarket_markets, kalshi_markets)
        
        for event_id, pm_price, kalshi_price in matched:
            # Opportunity 1: Buy cheap, sell expensive
            if pm_price < kalshi_price - self._min_edge(pm_price):
                profit = ((1.0 / pm_price) * kalshi_price - 1.0) * 100
                opportunities.append(ArbOpportunity(
                    event_id=event_id,
                    polymarket_price=pm_price,
                    kalshi_price=kalshi_price,
                    direction="buy_polymarket_sell_kalshi",
                    profit_per_contract=profit,
                    max_contracts=self.max_position
                ))
            elif kalshi_price < pm_price - self._min_edge(kalshi_price):
                profit = ((1.0 / kalshi_price) * pm_price - 1.0) * 100
                opportunities.append(ArbOpportunity(
                    event_id=event_id,
                    polymarket_price=pm_price,
                    kalshi_price=kalshi_price,
                    direction="buy_kalshi_sell_polymarket",
                    profit_per_contract=profit,
                    max_contracts=self.max_position
                ))
        
        return sorted(opportunities, key=lambda x: -x.profit_per_contract)
    
    def _match_identical_events(self, pm_markets, kalshi_markets):
        """
        Match markets across platforms using event title similarity
        and resolution criteria. Returns (event_id, pm_price, kalshi_price).
        """
        matches = []
        for pm in pm_markets:
            for k in kalshi_markets:
                if self._same_event(pm.title, k.title, pm.expiration, k.expiration):
                    matches.append((
                        pm.title,
                        pm.current_price,
                        k.current_price
                    ))
        return matches
    
    def _min_edge(self, price: float) -> float:
        """Minimum profitable edge considering fees on both platforms."""
        polymarket_fee = 0.001  # 0.1%
        kalshi_fee = 0.002      # 0.2%
        total_fee_pct = (polymarket_fee + kalshi_fee) * price
        return max(total_fee_pct, 0.005)  # At least 0.5%
    
    async def execute_arb(self, opp: ArbOpportunity):
        """Execute simultaneous trades on both platforms."""
        if opp.direction == "buy_polymarket_sell_kalshi":
            await self.polymarket.buy("YES", opp.polymarket_price, opp.max_contracts)
            await self.kalshi.buy("NO", 1 - opp.kalshi_price, opp.max_contracts)
        else:
            await self.kalshi.buy("YES", opp.kalshi_price, opp.max_contracts)
            await self.polymarket.buy("NO", 1 - opp.polymarket_price, opp.max_contracts)
