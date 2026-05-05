"""
NBA Market Microstructure Arbitrage — Liquidity-Adjusted Opportunity Scanner

Based on arXiv:2605.00864 (Cheng, Yang, Zou — UCLA, May 2026)
75M+ LOB snapshots across 173 NBA games.

Key findings baked into this prototype:
- Single-market arb is rare: only 7 episodes in 173 games
- Median single-market duration: 3.6 seconds
- Combinatorial arb: 290 episodes, median return 101 bps
- 76.9% of combinatorial opportunities limited to avg 14.8 shares

This scanner wraps a basic combinatorial detection with a liquidity filter
to estimate REAL achievable profit vs theoretical edge.
"""

import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketQuote:
    condition_id: str
    outcome: str  # 'YES' or 'NO'
    price: float
    shares_available: float  # depth at best price

@dataclass
class ArbOpportunity:
    market_a: MarketQuote
    market_b: MarketQuote
    theoretical_cost: float       # price_a + price_b
    theoretical_profit_bps: float # (1.0 - cost) / cost * 10000
    max_achievable_shares: float  # min(shares_a, shares_b)
    liquidity_adjusted_profit: float  # theoretical_profit * max_shares
    is_combinatorial: bool
    notes: str = ""

def calculate_arb(market_a: MarketQuote, market_b: MarketQuote) -> ArbOpportunity:
    """
    Detect and measure an arbitrage opportunity between two markets.
    """
    # Combinatorial = True if condition_ids differ (cross-market)
    is_combinatorial = market_a.condition_id != market_b.condition_id

    cost = market_a.price + market_b.price

    if cost >= 1.0:
        return ArbOpportunity(
            market_a=market_a, market_b=market_b,
            theoretical_cost=cost, theoretical_profit_bps=0.0,
            max_achievable_shares=0.0, liquidity_adjusted_profit=0.0,
            is_combinatorial=is_combinatorial,
            notes="No arb: cost >= $1.00"
        )

    profit_bps = (1.0 - cost) / cost * 10000
    max_shares = min(market_a.shares_available, market_b.shares_available)
    liquidity_adj_profit = profit_bps * max_shares / 10000  # profit per share in $

    notes = ""
    if is_combinatorial and max_shares < 15:
        notes = "⚠️ Retail-scale only (≤14.8 avg executable shares per Cheng et al.)"
    elif not is_combinatorial and profit_bps > 0:
        notes = "⚠️ Single-market arb is extremely rare (only 7 episodes in 173 games)"
    elif profit_bps >= 100:
        notes = "✅ Combinatorial arb with ≥100 bps return"

    return ArbOpportunity(
        market_a=market_a, market_b=market_b,
        theoretical_cost=cost,
        theoretical_profit_bps=round(profit_bps, 2),
        max_achievable_shares=max_shares,
        liquidity_adjusted_profit=round(liquidity_adj_profit, 4),
        is_combinatorial=is_combinatorial,
        notes=notes
    )

def filter_live_opportunities(opportunities: list[ArbOpportunity],
                               min_profit_bps: float = 10.0,
                               min_shares: float = 5.0) -> list[ArbOpportunity]:
    """
    Apply empirical thresholds from the NBA LOB study.
    - min_profit_bps: Only flag opportunities with real edge (default 10 bps = $0.001)
    - min_shares: Only flag if depth supports at least 5 shares (retail minimum)
    """
    filtered = []
    for arb in opportunities:
        if (arb.theoretical_profit_bps >= min_profit_bps
                and arb.max_achievable_shares >= min_shares):
            filtered.append(arb)
    return filtered

def estimate_realistic_profit(arb: ArbOpportunity,
                               historical_avg_return_bps: float = 101.0) -> float:
    """
    Adjust expected profit using empirical findings:
    - Combinatorial median return: 101 bps (not theoretical max)
    - Single-market median duration: 3.6 seconds (not tradeable without co-location)

    Returns realistic expected profit per share ($).
    """
    if arb.is_combinatorial:
        realistic_bps = min(arb.theoretical_profit_bps, historical_avg_return_bps)
    else:
        # Single-market: practically untradeable at scale
        realistic_bps = min(arb.theoretical_profit_bps, 50.0) * 0.5  # 50% of edge, derated

    return realistic_bps * arb.max_achievable_shares / 10000

# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulated Polymarket NBA market quotes (combinatorial pair example)
    # "Team A wins game" (condition: game_winner) + "Team A leads at halftime" (condition: halftime)
    game_winner_yes = MarketQuote(
        condition_id="game_123_winner",
        outcome="YES",
        price=0.47,
        shares_available=12.0
    )
    halftime_lead_no = MarketQuote(
        condition_id="game_123_halftime",
        outcome="NO",
        price=0.50,
        shares_available=18.0
    )

    arb = calculate_arb(game_winner_yes, halftime_lead_no)
    print(f"Cost: ${arb.theoretical_cost:.2f}  |  Edge: {arb.theoretical_profit_bps:.1f} bps")
    print(f"Max shares: {arb.max_achievable_shares:.0f}  |  {arb.notes}")
    print(f"Realistic profit (derated): ${estimate_realistic_profit(arb):.4f} per share")

    # Single-market example
    single_yes = MarketQuote("single_001", "YES", 0.43, 8.0)
    single_no  = MarketQuote("single_001", "NO",  0.52, 8.0)
    arb2 = calculate_arb(single_yes, single_no)
    print(f"\nSingle-market: Cost ${arb2.theoretical_cost:.2f} | Edge {arb2.theoretical_profit_bps:.1f} bps | {arb2.notes}")
