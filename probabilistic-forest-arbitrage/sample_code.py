"""
Reference implementation: Integer programming arbitrage detection
(inspired by Saguillo et al. 2025 "Unravelling the Probabilistic Forest")
"""

from itertools import combinations
from typing import NamedTuple


class ArbitrageOpportunity(NamedTuple):
    """An arbitrage opportunity on Polymarket."""
    market_ids: list[str]
    condition_ids: list[str]
    positions: list[float]  # buy prices
    guaranteed_payout: float  # $1.00 for binary markets
    profit_per_unit: float
    type: str  # "rebalancing" or "combinatorial"


def detect_single_market_arb(markets: list[dict]) -> list[ArbitrageOpportunity]:
    """
    Detect single-market (rebalancing) arbitrage.
    YES + NO should sum to ~$1.00.
    If YES_price + NO_price < $1.00, buying both guarantees profit.
    """
    opportunities = []
    for m in markets:
        yes_price = m.get("yes_bid", 0)
        no_price = m.get("no_bid", 0)
        total = yes_price + no_price
        if total < 0.99 and yes_price > 0 and no_price > 0:
            profit = 1.0 - total
            opportunities.append(
                ArbitrageOpportunity(
                    market_ids=[m["id"]],
                    condition_ids=[m["condition_id"]],
                    positions=[yes_price, no_price],
                    guaranteed_payout=1.0,
                    profit_per_unit=profit,
                    type="rebalancing",
                )
            )
    return sorted(opportunities, key=lambda o: o.profit_per_unit, reverse=True)


def detect_combinatorial_arb(
    conditions: dict[str, list[dict]]
) -> list[ArbitrageOpportunity]:
    """
    Detect combinatorial arbitrage across related conditions.
    When logically dependent markets (e.g., "BTC > 100k" and "BTC > 90k")
    have prices that violate logical consistency, arb exists.
    """
    opportunities = []

    for condition_id, markets in conditions.items():
        for m1, m2 in combinations(markets, 2):
            # Check for logical nesting (e.g., event A is subset of event B)
            if not _are_logically_nested(m1, m2):
                continue

            # Example: if P(BTC>100k) > P(BTC>90k), arb exists
            # since BTC>100k is a subset of BTC>90k
            p1 = m1.get("price", 0)
            p2 = m2.get("price", 0)

            if m1.get("is_subset_of") == m2["id"] and p1 > p2:
                profit = p1 - p2
                opportunities.append(
                    ArbitrageOpportunity(
                        market_ids=[m1["id"], m2["id"]],
                        condition_ids=[condition_id, condition_id],
                        positions=[p1, p2],
                        guaranteed_payout=1.0,
                        profit_per_unit=profit,
                        type="combinatorial",
                    )
                )

    return sorted(opportunities, key=lambda o: o.profit_per_unit, reverse=True)


def _are_logically_nested(m1: dict, m2: dict) -> bool:
    """Check if two markets have a logical subset relationship."""
    return (
        m1.get("is_subset_of") == m2.get("id")
        or m2.get("is_subset_of") == m1.get("id")
    )
