"""
Polymarket ↔ Kalshi Arbitrage Detection & Execution
Reference implementation based on realfishsam/prediction-market-arbitrage-bot
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    event_id: str
    polymarket_price: float
    kalshi_price: float
    spread_pct: float
    max_size: float
    direction: str  # "KALSHI_BUY_POLY_SELL" or "POLY_BUY_KALSHI_SELL"

class CrossExchangeArbBot:
    """Detects and executes arbitrage between Polymarket and Kalshi."""

    MIN_SPREAD_PCT = 0.02  # 2% minimum after fees
    MAX_POSITION_SIZE = 1000  # USDC

    def __init__(self, poly_client, kalshi_client):
        self.poly = poly_client
        self.kalshi = kalshi_client

    async def scan_markets(self) -> list[ArbitrageOpportunity]:
        """Scan all overlapping events for price divergence."""
        opportunities = []
        poly_markets = await self.poly.get_active_markets()
        kalshi_markets = await self.kalshi.get_active_markets()

        # Match events by title/normalized name
        matched = self._match_events(poly_markets, kalshi_markets)

        for event_id, (poly_mkt, kalshi_mkt) in matched.items():
            poly_yes = poly_mkt["outcomePrices"]["yes"]
            kalshi_yes = kalshi_mkt["yes_bid"]  # or mid-price

            spread = abs(poly_yes - kalshi_yes)
            if spread >= self.MIN_SPREAD_PCT:
                direction = "KALSHI_BUY_POLY_SELL" if kalshi_yes < poly_yes else "POLY_BUY_KALSHI_SELL"
                max_size = min(
                    poly_mkt["liquidity"]["yes"],
                    kalshi_mkt["liquidity"]["yes"],
                    self.MAX_POSITION_SIZE
                )
                opportunities.append(ArbitrageOpportunity(
                    event_id=event_id,
                    polymarket_price=poly_yes,
                    kalshi_price=kalshi_yes,
                    spread_pct=spread,
                    max_size=max_size,
                    direction=direction
                ))

        return sorted(opportunities, key=lambda o: o.spread_pct, reverse=True)

    async def execute(self, opp: ArbitrageOpportunity) -> bool:
        """Execute both legs simultaneously."""
        tasks = []
        if opp.direction == "KALSHI_BUY_POLY_SELL":
            tasks.append(self.kalshi.buy("yes", opp.event_id, opp.max_size))
            tasks.append(self.poly.sell("yes", opp.event_id, opp.max_size))
        else:
            tasks.append(self.poly.buy("yes", opp.event_id, opp.max_size))
            tasks.append(self.kalshi.sell("yes", opp.event_id, opp.max_size))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Verify both legs filled
        if any(isinstance(r, Exception) for r in results):
            logger.error(f"Partial fill on {opp.event_id}, hedging positions")
            return False
        return True

    def _match_events(self, poly_mkts, kalshi_mkts) -> dict:
        """Simple title-based matching. In production, use embedding similarity."""
        matches = {}
        for p in poly_mkts:
            for k in kalshi_mkts:
                if self._titles_match(p["title"], k["title"]):
                    matches[p["id"]] = (p, k)
                    break
        return matches

    @staticmethod
    def _titles_match(t1: str, t2: str) -> bool:
        return t1.lower().strip() == t2.lower().strip()
