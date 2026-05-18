"""
PolyHFT — Combinatorial Arbitrage Strategy (Simplified)

Detects cross-market arbitrage by finding logical inconsistencies
between related prediction market outcomes.
"""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass
class Market:
    condition_id: str
    token_id: str
    outcome: str  # "YES" or "NO"
    price: float  # cents (0-100)


@dataclass
class ArbitrageOpportunity:
    markets: List[Market]
    total_cost: float
    expected_payout: float
    profit: float
    edge_pct: float


class CombinatorialArbDetector:
    """
    Detects arbitrage opportunities across logically related prediction markets.
    
    Key insight: If outcomes A and B are mutually exclusive and collectively exhaustive,
    then P(A) + P(B) should equal 1.0. Any deviation > fees represents an opportunity.
    """

    def __init__(self, min_edge_pct: float = 0.02):
        self.min_edge_pct = min_edge_pct
        self.gamma_api = "https://gamma-api.polymarket.com"

    async def fetch_order_book(self, session: aiohttp.ClientSession, 
                                condition_id: str) -> Dict:
        """Fetch order book for a given condition/market."""
        url = f"https://clob.polymarket.com/orderbook/{condition_id}"
        async with session.get(url) as resp:
            return await resp.json()

    def detect_bundle_arb(self, yes_price: float, no_price: float) -> float:
        """
        Bundle arbitrage: YES + NO should equal ~$1.00.
        
        If YES = 0.47 and NO = 0.40, total = 0.87.
        Buying both guarantees $1.00 at resolution = $0.13 profit.
        """
        total = yes_price + no_price
        if total < 1.0:
            return 1.0 - total  # profit per share
        return 0.0

    def detect_cross_condition_arb(self, markets: List[Market]) -> List[ArbitrageOpportunity]:
        """
        Cross-condition arbitrage: Find portfolios of tokens across conditions
        that sum to less than $1.00.
        
        Example: 
        - ETH Up = $0.47
        - BTC Down = $0.40
        Total = $0.87 → guaranteed $1.00 at resolution = $0.13 profit
        
        Since the events are independent, at least one outcome in each
        pair will resolve to $1.00.
        """
        opportunities = []
        
        # Group by condition
        conditions: Dict[str, List[Market]] = {}
        for m in markets:
            if m.condition_id not in conditions:
                conditions[m.condition_id] = []
            conditions[m.condition_id].append(m)
        
        # Find combinations where one token from each condition
        # sums to less than $1.00
        condition_list = list(conditions.values())
        
        def combine(idx: int, selected: List[Market]):
            if idx == len(condition_list):
                if len(selected) == len(condition_list):
                    total_cost = sum(m.price for m in selected)
                    if total_cost < 1.0 - self.min_edge_pct:
                        opp = ArbitrageOpportunity(
                            markets=selected,
                            total_cost=total_cost,
                            expected_payout=1.0,
                            profit=1.0 - total_cost,
                            edge_pct=(1.0 - total_cost) * 100,
                        )
                        opportunities.append(opp)
                return
            
            for m in condition_list[idx]:
                combine(idx + 1, selected + [m])
        
        combine(0, [])
        
        # Sort by profit descending
        opportunities.sort(key=lambda x: x.profit, reverse=True)
        return opportunities

    async def scan_markets(self, condition_ids: List[str]) -> List[ArbitrageOpportunity]:
        """Main entry point: scan all specified conditions for arbitrage."""
        async with aiohttp.ClientSession() as session:
            markets = []
            for cid in condition_ids:
                ob = await self.fetch_order_book(session, cid)
                if not ob or "bids" not in ob or "asks" not in ob:
                    continue
                
                # Get best bid/ask
                best_bid = float(ob["bids"][0]["price"]) if ob["bids"] else 0
                best_ask = float(ob["asks"][0]["price"]) if ob["asks"] else 1
                
                markets.append(Market(
                    condition_id=cid,
                    token_id="",
                    outcome="YES",
                    price=best_ask,
                ))
                markets.append(Market(
                    condition_id=cid,
                    token_id="",
                    outcome="NO",
                    price=1.0 - best_bid,
                ))
            
            return self.detect_cross_condition_arb(markets)


async def main():
    detector = CombinatorialArbDetector(min_edge_pct=0.02)
    
    # Example: Scan BTC and ETH 15-minute conditions
    condition_ids = [
        # These would be actual condition IDs from Polymarket
        "btc-15m-up-condition-id",
        "btc-15m-down-condition-id",
        "eth-15m-up-condition-id",
        "eth-15m-down-condition-id",
    ]
    
    opportunities = await detector.scan_markets(condition_ids)
    
    for opp in opportunities[:5]:
        outcomes = [f"{m.outcome} @ ${m.price:.2f}" for m in opp.markets]
        print(f"ARB: {' + '.join(outcomes)}")
        print(f"  Cost: ${opp.total_cost:.2f} → Payout: ${opp.expected_payout:.2f}")
        print(f"  Profit: ${opp.profit:.2f} ({opp.edge_pct:.1f}%)")
        print()


if __name__ == "__main__":
    asyncio.run(main())
