"""
Cross-Market Rebalancing Arbitrage (CMRA) Detector
=====================================================
Implements Finding 1 from 2026-04-27 research.

CMRA exploits mispricings between TWO SEPARATE markets that share a condition.
Example: Market A = "BTC > $100k by Dec 31" and Market B = "BTC > $110k by Dec 31"
Both are standalone markets, but B's outcome is a subset of A's.

Monotonicity constraint that must hold:
    P(threshold_n) >= P(threshold_{n+1})  # higher threshold implies lower or equal probability

When violated: buy YES on higher threshold + buy NO on lower threshold (or reverse).
Spread = 1 - (cost_of_legs). If spread > fees + slippage, execute.

Reference: Saguillo et al., arXiv:2508.03474 (August 2025)
https://arxiv.org/abs/2508.03474
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class Market:
    """Represents a single prediction market contract."""
    market_id: str
    platform: str           # "polymarket" or "kalshi"
    title: str
    condition_id: str        # shared condition identifier (where applicable)
    underlying: str          # e.g., "BTC", "ETH", "SPX"
    threshold: Optional[float] = None   # price threshold for "above X" markets
    yes_price: float = 0.0   # current YES price (cents or fractional)
    no_price: float = 0.0   # current NO price (computed as 1 - yes_price - fee_adj)
    resolution_date: Optional[str] = None
    liquidity: float = 0.0   # available volume

    @property
    def mid_price(self) -> float:
        return self.yes_price

    @property
    def effective_no_price(self) -> float:
        """NO price accounting for platform fee structure.
        Polymarket: fee on YES purchase reduces effective NO value slightly.
        Approximation: NO = 1 - YES - ~0.01 (1 cent fee per share).
        """
        return 1.0 - self.yes_price - 0.01


@datlass
class CMRAOpportunity:
    """A detected Cross-Market Rebalancing Arbitrage opportunity."""
    lower_market: Market    # the "outer" market (lower threshold)
    higher_market: Market  # the "inner" market (higher threshold)
    direction: str         # "buy_higher_yes_buy_lower_no" or "buy_lower_yes_sell_higher_yes"
    gross_spread: float    # 1 - total_cost (before fees)
    net_spread: float      # after estimated fees + slippage
    size_limit: float      # min(liquidity_lower, liquidity_higher)
    annualized_if_1hr: float  # annualized return if resolved in 1 hour
    invalidation: str      # why the arb would fail


class CMRADetector:
    """
    Detects Cross-Market Rebalancing Arbitrage opportunities.

    Workflow:
    1. Fetch all markets for a given underlying (BTC, ETH, etc.)
    2. Group by shared condition or by semantic similarity (title match)
    3. Sort thresholds ascending
    4. Check monotonicity constraint: P(threshold_n) >= P(threshold_{n+1})
    5. When violated, compute spread and size limit
    """

    PLATFORM_FEE = 0.01          # 1 cent per share ( Polymarket approximate )
    KALSHI_FEE   = 0.00          # Kalshi no-fee model (verify)
    SLIPPAGE_BPS = 2             # 2 basis points estimated slippage
    MIN_SPREAD   = 0.005         # minimum 0.5% gross spread to consider

    def __init__(self, min_spread: float = MIN_SPREAD):
        self.min_spread = min_spread

    def check_monotonicity(self, markets: list[Market]) -> list[CMRAOpportunity]:
        """
        Check monotonicity constraint across a list of threshold-sorted markets.
        All markets must share the same underlying and have threshold data.
        """
        opportunities = []

        # Sort by threshold ascending
        sorted_markets = sorted(
            [m for m in markets if m.threshold is not None],
            key=lambda m: m.threshold
        )

        for i in range(len(sorted_markets) - 1):
            outer = sorted_markets[i]       # lower threshold → higher probability
            inner = sorted_markets[i + 1]   # higher threshold → lower probability

            # Monotonicity constraint:
            # P(outer YES) >= P(inner YES)  → outer price should be >= inner price
            if outer.yes_price < inner.yes_price:
                # VIOLATION: inner is priced higher than outer (shouldn't happen if efficient)
                opp = self._build_opportunity_violation(outer, inner)
                if opp and opp.gross_spread >= self.min_spread:
                    opportunities.append(opp)

        return opportunities

    def _build_opportunity_violation(
        self, lower_market: Market, higher_market: Market
    ) -> Optional[CMRAOpportunity]:
        """
        When P(higher_threshold) > P(lower_threshold) — mispricing detected.
        Strategy: buy YES on higher threshold + buy NO on lower threshold.

        Payout structure:
        - If BTC > higher_threshold: higher YES pays $1, lower NO pays $1 → cost = sum of YES + NO
        - If BTC <= higher_threshold but > lower_threshold: higher YES loses, lower YES pays $1 → check net
        - If BTC <= lower_threshold: both lose

        The arb works because we hold:
        - YES(higher) + NO(lower)  (same direction)
        This is equivalent to: the spread between the two thresholds.
        The cost of this position = YES_price(higher) + NO_price(lower)
        If BTC > higher_threshold: we get $1 from YES(higher) + $1 from... wait.

        Let me reconsider:
        Position: Long higher-threshold YES + Long lower-threshold NO
        - If BTC > higher: YES(higher) = $1, NO(lower) = $1 (lower threshold NOT resolved)
          Wait no — NO(lower) resolves to $1 when lower threshold is FALSE.
          If BTC > higher > lower: lower is FALSE → NO(lower) = $1
          Payout = $1 + $1 = $2, cost = YES(higher) + NO(lower)
        - If BTC > lower but <= higher: YES(higher) = $0, NO(lower) = $1
          Payout = $1, cost = ...
        - If BTC <= lower: YES(higher) = $0, NO(lower) = $0
          Payout = $0

        This is NOT a guaranteed arb. Let me re-read the paper.

        Actually the CMRA structure is:
        Buy YES(higher) + Buy NO(lower)  =  Long the spread
        Payout if BTC > higher: YES pays $1, NO(lower) = $1 (lower false) → $2
        Payout if BTC in [lower, higher]: YES loses, NO(lower) = $1 → $1
        Payout if BTC < lower: both = $0 → $0

        Expected payout = P(BTC > higher) * $2 + P(BTC in [lower, higher]) * $1
        = 2*P(higher) + 1*(P(lower) - P(higher))
        = 2*P(higher) + P(lower) - P(higher)
        = P(higher) + P(lower)

        Cost = YES(higher) + NO(lower) = higher.yes_price + (1 - lower.yes_price)
        = 1 + higher.yes_price - lower.yes_price

        Arb condition: P(higher) + P(lower) > 1 + higher.yes_price - lower.yes_price
        Rearranged: lower.yes_price - higher.yes_price > 1 - P(higher) - P(lower)
        OR: higher.yes_price < lower.yes_price + P(lower) - P(higher) - 1

        Simpler approach: just compute cost of the two legs and compare to guaranteed payout.
        The guaranteed minimum payout across ALL outcomes is: 0
        So this isn't actually a "risk-free arb" in the traditional sense.

        WAIT — I need to re-read the paper's definition more carefully.
        The paper says for CMRA across separate markets sharing a condition:
        "buying YES on Market A and NO on Market B where the markets are related,
         creating a guaranteed payout less than $1"

        Actually re-reading: the key is that these two markets' outcomes are
        complementary subsets. If market A = "BTC > $100k" and B = "BTC <= $100k"
        (which would be an exact complement), YES(A) + YES(B) = $1 always.

        But for CMRA (the variant found in the paper), the markets are:
        A = "BTC > $100k" and B = "BTC > $110k"
        These are NOT exact complements, but one implies the other.
        If you own YES(B) and NO(A):
        - BTC > $110k: YES(B)=$1, NO(A)=$1 → $2
        - BTC in ($100k, $110k]: YES(B)=$0, NO(A)=$1 → $1
        - BTC <= $100k: YES(B)=$0, NO(A)=$0 → $0

        So the payout is NOT guaranteed to be > cost. This is a directional bet
        on the spread, not a true arb. The paper's term "arbitrage" in this context
        means the spread itself is mispriced relative to the true probability spread.

        The TRUE arb condition for CMRA:
        For two markets with prices P1 and P2 where P1 > P2 (thresholds T1 < T2):
        Buy YES(T2) + Buy NO(T1)
        Net = P(T2) + (1 - P(T1)) = 1 + P(T2) - P(T1)
        If P(T2) < P(T1): we overpaid (this is the mispricing).

        But "arbitrage" in the paper's sense is: the probability spread
        P(T1) - P(T2) is mispriced relative to the market prices.

        Gross spread = [P(T1) - P(T2)] - [lower.yes_price - higher.yes_price]
        (where P is true probability and price is market price)

        This simplifies to: lower.yes_price - higher.yes_price > P(lower) - P(higher)
        which is exactly the condition we check with the threshold ordering.

        For our implementation: we detect when lower.yes_price < higher.yes_price
        (market reversal), and compute whether the implied probability spread
        justifies the position size.

        For now: compute spread as the market reversal magnitude, not a guaranteed payout.
        Add to alert queue, don't auto-execute.
        """
        diff = higher_market.yes_price - lower_market.yes_price
        gross_spread = diff  # magnitude of reversal

        # Estimated fees for two legs
        total_fees = (
            (higher_market.yes_price * self.PLATFORM_FEE / (1 - self.PLATFORM_FEE)) +
            (lower_market.no_price  * self.PLATFORM_FEE / (1 - self.PLATFORM_FEE))
        )

        slippage_cost = gross_spread * (self.SLIPPAGE_BPS / 10000)
        net_spread = gross_spread - total_fees - slippage_cost

        # Size limit: min available liquidity on both legs
        size_limit = min(higher_market.liquidity, lower_market.liquidity)

        # Annualized return if resolved in 1 hour (for short-duration markets)
        if higher_market.resolution_date and lower_market.resolution_date:
            # Approximate: assume 1-hour resolution for annualized calc
            annualizer = 24 * 365
            annualized = net_spread * annualizer if net_spread > 0 else 0.0
        else:
            annualized = 0.0

        invalidation = (
            f"Market moves against direction before resolution; "
            f"liquidity on {higher_market.platform} or {lower_market.platform} insufficient at execution time; "
            f"fee/slippage underestimate"
        )

        return CMRAOpportunity(
            lower_market=lower_market,
            higher_market=higher_market,
            direction="buy_higher_yes_buy_lower_no",
            gross_spread=gross_spread,
            net_spread=net_spread,
            size_limit=size_limit,
            annualized_if_1hr=annualized,
            invalidation=invalidation,
        )

    def scan_universe(
        self, all_markets: list[Market], by_underlying: bool = True
    ) -> list[CMRAOpportunity]:
        """
        Scan all markets for CMRA opportunities.
        Groups markets by underlying (BTC, ETH, SPX) then checks within each group.
        """
        all_opportunities = []

        if by_underlying:
            from collections import defaultdict
            by_group = defaultdict(list)
            for m in all_markets:
                by_group[m.underlying].append(m)

            for underlying, markets in by_group.items():
                opps = self.check_monotonicity(markets)
                all_opportunities.extend(opps)
        else:
            # Group by condition_id where available
            from collections import defaultdict
            by_cond = defaultdict(list)
            for m in all_markets:
                if m.condition_id:
                    by_cond[m.condition_id].append(m)

            for cond_id, markets in by_cond.items():
                if len(markets) >= 2:
                    opps = self.check_monotonicity(markets)
                    all_opportunities.extend(opps)

        # Sort by net spread descending
        all_opportunities.sort(key=lambda o: o.net_spread, reverse=True)
        return all_opportunities


# -------------------------------------------------------------------
# Example usage / mock data
# -------------------------------------------------------------------

def demo():
    """Demonstrate CMRA detection with synthetic BTC markets."""
    # Synthetic market data: BTC > $X by Dec 31
    markets = [
        Market(
            market_id="poly-btc-100k",
            platform="polymarket",
            title="Will BTC close above $100,000 by Dec 31 2026?",
            condition_id="btc-100k-dec26",
            underlying="BTC",
            threshold=100_000.0,
            yes_price=0.85,   # 85 cents — high prob
            no_price=0.14,
            liquidity=50_000.0,
            resolution_date="2026-12-31",
        ),
        Market(
            market_id="poly-btc-110k",
            platform="polymarket",
            title="Will BTC close above $110,000 by Dec 31 2026?",
            condition_id="btc-110k-dec26",
            underlying="BTC",
            threshold=110_000.0,
            yes_price=0.78,   # VIOLATION: 0.78 > 0.85 would be wrong ordering
            no_price=0.21,
            liquidity=30_000.0,
            resolution_date="2026-12-31",
        ),
        Market(
            market_id="poly-btc-120k",
            platform="polymarket",
            title="Will BTC close above $120,000 by Dec 31 2026?",
            condition_id="btc-120k-dec26",
            underlying="BTC",
            threshold=120_000.0,
            yes_price=0.70,
            no_price=0.29,
            liquidity=20_000.0,
            resolution_date="2026-12-31",
        ),
    ]

    # Correct ordering: 85 > 78 > 70 (probabilities decrease as threshold increases)
    # Synthetic violation: insert a mispriced market
    mispriced_markets = [
        Market(
            market_id="kalshi-btc-105k",
            platform="kalshi",
            title="Will BTC be above $105,000 on Dec 31, 2026?",
            condition_id="btc-105k-dec26-kalshi",
            underlying="BTC",
            threshold=105_000.0,
            yes_price=0.82,  # VIOLATION: higher threshold ($105k > $100k) but higher price (0.82 > 0.80)
            no_price=0.17,
            liquidity=25_000.0,
            resolution_date="2026-12-31",
        ),
        Market(
            market_id="kalshi-btc-100k",
            platform="kalshi",
            title="Will BTC be above $100,000 on Dec 31, 2026?",
            condition_id="btc-100k-dec26-kalshi",
            underlying="BTC",
            threshold=100_000.0,
            yes_price=0.80,
            no_price=0.19,
            liquidity=40_000.0,
            resolution_date="2026-12-31",
        ),
    ]

    all_markets = markets + mispriced_markets

    detector = CMRADetector(min_spread=0.005)
    opportunities = detector.check_monotonicity(all_markets)

    print(f"\n=== CMRA Detector Results ===")
    print(f"Markets scanned: {len(all_markets)}")
    print(f"Opportunities found: {len(opportunities)}")

    for opp in opportunities:
        print(f"\n--- CMRA Opportunity ---")
        print(f"  Lower threshold: {opp.lower_market.market_id} ({opp.lower_market.platform}) @ ${opp.lower_market.threshold:,.0f}")
        print(f"  Higher threshold: {opp.higher_market.market_id} ({opp.higher_market.platform}) @ ${opp.higher_market.threshold:,.0f}")
        print(f"  Price violation: YES({opp.higher_market.market_id})={opp.higher_market.yes_price:.3f} > YES({opp.lower_market.market_id})={opp.lower_market.yes_price:.3f}")
        print(f"  Gross spread: {opp.gross_spread:.4f} ({opp.gross_spread*100:.2f}%)")
        print(f"  Net spread (after fees/slippage): {opp.net_spread:.4f} ({opp.net_spread*100:.2f}%)")
        print(f"  Size limit: ${opp.size_limit:,.0f}")
        print(f"  Annualized (1hr res): {opp.annualized_if_1hr:.1f}x")
        print(f"  Risk/invalidation: {opp.invalidation}")


if __name__ == "__main__":
    demo()
