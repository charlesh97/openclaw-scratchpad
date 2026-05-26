"""
Reference implementation: Cross-platform arbitrage detection pattern
from the Polymarket-Kalshi Arbitrage Bot.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ArbitrageOpportunity:
    polymarket_price: float  # cost of position on Polymarket (cents)
    kalshi_price: float  # cost of opposite position on Kalshi (cents)
    total_cost: float  # combined cost
    edge: float  # profit per $1 unit
    polymarket_market_id: str
    kalshi_market_id: str
    direction: str  # "poly_down_kalshi_yes" or "poly_up_kalshi_no"


class ArbitrageScanner:
    """
    Scans Polymarket and Kalshi for price discrepancies.
    Core detection logic — not the full bot implementation.
    """

    MIN_EDGE = 0.01  # 1% minimum edge
    FEE_RATE = 0.002  # 0.2% platform + gas fee estimate

    def find_opportunities(
        self, polymarket_prices: dict, kalshi_prices: dict
    ) -> list[ArbitrageOpportunity]:
        opportunities = []

        for event_id, poly_data in polymarket_prices.items():
            if event_id not in kalshi_prices:
                continue

            kalshi_data = kalshi_prices[event_id]

            # Strategy 1: Poly DOWN + Kalshi YES
            poly_down = poly_data.get("down_price", 0)
            kalshi_yes = kalshi_data.get("yes_price", 0)
            total_1 = poly_down + kalshi_yes
            edge_1 = 1.0 - total_1 - self.FEE_RATE

            if edge_1 >= self.MIN_EDGE:
                opportunities.append(
                    ArbitrageOpportunity(
                        polymarket_price=poly_down,
                        kalshi_price=kalshi_yes,
                        total_cost=total_1,
                        edge=edge_1,
                        polymarket_market_id=poly_data["market_id"],
                        kalshi_market_id=kalshi_data["market_id"],
                        direction="poly_down_kalshi_yes",
                    )
                )

            # Strategy 2: Poly UP + Kalshi NO
            poly_up = poly_data.get("up_price", 0)
            kalshi_no = kalshi_data.get("no_price", 0)
            total_2 = poly_up + kalshi_no
            edge_2 = 1.0 - total_2 - self.FEE_RATE

            if edge_2 >= self.MIN_EDGE:
                opportunities.append(
                    ArbitrageOpportunity(
                        polymarket_price=poly_up,
                        kalshi_price=kalshi_no,
                        total_cost=total_2,
                        edge=edge_2,
                        polymarket_market_id=poly_data["market_id"],
                        kalshi_market_id=kalshi_data["market_id"],
                        direction="poly_up_kalshi_no",
                    )
                )

        return sorted(opportunities, key=lambda o: o.edge, reverse=True)
