#!/usr/bin/env python3
"""
Combinatorial Arbitrage Detection — vega research
==================================================
Concept: Detect mispricings across MULTIPLE RELATED but separate markets.

arxiv:2508.03474 ("Unravelling the Probabilistic Forest") identifies two types:
  1. Market Rebalancing Arbitrage (MRA) = YES+NO sum < 1 within single contract
     → covered by arb-bot-main check_parity()
  2. Combinatorial Arbitrage (CA) = price violations across related but SEPARATE
     markets where one outcome implies another

Example:
  Market A: "BTC > $100k by Dec 31 2026"    → YES = 0.70
  Market B: "BTC > $110k by Dec 31 2026"    → YES = 0.60
  Problem: If BTC > $110k then BTC > $100k, so B implies A.
           Therefore P(B) <= P(A) must hold (monotonicity constraint).
           Here 0.60 < 0.70 — VIOLATION: sell A, buy B = risk-free arb.

This script demonstrates:
  1. Market clustering by topical similarity (keyword/embedding)
  2. Filtering by shared underlying + temporal overlap
  3. Combinatorial constraint checking (nested conditions)
  4. Opportunity scoring

Note: O(2^n) reduction via heuristic filtering is essential.
"""

from dataclasses import dataclass
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Market:
    id: str
    question: str           # raw question text
    platform: str           # "kalshi" | "polymarket"
    yes_bid: float          # best bid (we buy at ask, sell at bid)
    yes_ask: float         # best ask
    no_bid: float
    no_ask: float
    end_date_iso: str       # ISO timestamp
    market_id: str          # Kalshi market_id
    series_ticker: str      # e.g. "BTC.USDT.USDC" or "POLITICS"
    tags: list[str]         # e.g. ["btc", "price", "above"]
    condition_id: str       # Polymarket condition_id


@datlass
class CombinatorialOpportunity:
    outer_market: Market    # the "implied" market (B implies A)
    inner_market: Market    # the "implying" market (B → A)
    direction: str          # "buy_inner_sell_outer" | "buy_outer_sell_inner"
    implied_prob_ratio: float  # P(inner) / P(outer) — violation magnitude
    edge_per_dollar: float     # profit per $1 invested
    confidence: float           # 0–1, based on constraint clarity


# ---------------------------------------------------------------------------
# Step 1: Market clustering by shared underlying
# ---------------------------------------------------------------------------

def build_market_clusters(markets: list[Market], max_days_apart: int = 30) -> dict[str, list[Market]]:
    """
    Group markets by shared underlying using ticker + keyword tags.
    Returns dict: cluster_key → list[Market]
    """
    clusters: dict[str, list[Market]] = {}

    for m in markets:
        # Cluster key: ticker + first few tags to group by asset/event
        key = _cluster_key(m)
        clusters.setdefault(key, []).append(m)

    # Within each cluster, filter to those with overlapping time windows
    filtered: dict[str, list[Market]] = {}
    for key, cluster_markets in clusters.items():
        filtered[key] = _filter_temporal_overlap(cluster_markets, max_days_apart)

    return filtered


def _cluster_key(m: Market) -> str:
    """Build a rough cluster key from series_ticker + key tags."""
    tags = set(m.tags)
    # Common asset tags
    asset_tags = {"btc", "bitcoin", "eth", "ethereum", "sp500", "nasdaq",
                  "football", "nba", "nfl", "politics", "election", "weather"}
    core_tags = tags & asset_tags
    if not core_tags:
        core_tags = {m.series_ticker.split(".")[0]} if m.series_ticker else {"misc"}
    return "|".join(sorted(core_tags))


def _filter_temporal_overlap(markets: list[Market], max_days: int) -> list[Market]:
    """Keep only markets whose end_dates are within max_days of each other."""
    # Simple approach: group by month-year of end_date
    by_month: dict[str, list[Market]] = {}
    for m in markets:
        # Use YYYY-MM prefix of end_date_iso
        prefix = m.end_date_iso[:7] if m.end_date_iso else "unknown"
        by_month.setdefault(prefix, []).append(m)

    result: list[Market] = []
    for month_markets in by_month.values():
        result.extend(month_markets)
    return result


# ---------------------------------------------------------------------------
# Step 2: Combinatorial constraint detection
# ---------------------------------------------------------------------------

def check_combinatorial_constraints(cluster: list[Market]) -> list[CombinatorialOpportunity]:
    """
    For a cluster of related markets, find nested/monotonicity violations.
    
    Nested condition types handled:
      - Threshold nesting:  "BTC > $X" nested in "BTC > $Y" if Y > X
      - Temporal nesting:   "by date A" nested in "by date B" if A > B
      - Set inclusion:      "team A wins" nested in "team A top 3"
    
    Violation: P(inner) must <= P(outer) for nested conditions.
               If P(inner) > P(outer), buy outer, sell inner = guaranteed arb.
    """
    opportunities = []
    n = len(cluster)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            opp = _evaluate_pair(cluster[i], cluster[j])
            if opp is not None and opp.edge_per_dollar > 0.01:
                opportunities.append(opp)

    # Deduplicate (A,B) and (B,A) pairs
    return _deduplicate_opportunities(opportunities)


def _evaluate_pair(m_a: Market, m_b: Market) -> Optional[CombinatorialOpportunity]:
    """
    Evaluate whether m_a and m_b form a combinatorial violation.
    
    Strategy: check threshold monotonicity for same underlying.
    - m_a "outer": higher threshold or longer time horizon
    - m_b "inner": lower threshold or shorter time horizon
    
    Constraint: P(inner) <= P(outer) must hold.
    Violation: P(inner) > P(outer) → buy outer, sell inner = guaranteed profit.
    """
    # Detect direction: which is the "inner" (more specific) condition?
    direction = _detect_nesting_direction(m_a, m_b)
    if direction is None:
        return None

    if direction == "a_outer_b_inner":
        outer, inner = m_a, m_b
    else:
        outer, inner = m_b, m_a

    # Compute prices (normalize to 0-1 for Polymarket, 0-100¢ → 0-1)
    p_outer = _normalize_price(outer.yes_ask)
    p_inner = _normalize_price(inner.yes_ask)

    # Constraint: p_inner <= p_outer
    if p_inner > p_outer:
        # Violation magnitude
        implied_ratio = p_inner / p_outer
        edge_per_dollar = p_inner - p_outer  # buy outer, sell inner
        
        return CombinatorialOpportunity(
            outer_market=outer,
            inner_market=inner,
            direction="buy_outer_sell_inner",
            implied_prob_ratio=implied_ratio,
            edge_per_dollar=edge_per_dollar,
            confidence=_constraint_confidence(outer, inner),
        )
    
    return None


def _detect_nesting_direction(m_a: Market, m_b: Market) -> Optional[str]:
    """
    Detect if one market's condition is nested within the other.
    Returns "a_outer_b_inner" if A implies B (A is outer/more general),
            "b_outer_a_inner" if B implies A,
            None if no nesting relationship detected.
    """
    # --- Threshold nesting ---
    # Parse thresholds from question text (simple keyword approach)
    thresh_a = _extract_threshold(m_a.question)
    thresh_b = _extract_threshold(m_b.question)
    
    if thresh_a is not None and thresh_b is not None:
        if thresh_a > thresh_b:
            # A has higher threshold = more specific = inner, B is outer
            return "b_outer_a_inner"
        elif thresh_b > thresh_a:
            return "a_outer_b_inner"

    # --- Same question family (shared tags) — temporal nesting ---
    shared_tags = set(m_a.tags) & set(m_b.tags)
    if shared_tags and m_a.end_date_iso != m_b.end_date_iso:
        # Earlier end date = more specific = inner
        if m_a.end_date_iso < m_b.end_date_iso:
            return "b_outer_a_inner"  # A expires sooner = inner
        else:
            return "a_outer_b_inner"  # B expires sooner = inner

    # --- Same series ticker + overlapping times ---
    if (m_a.series_ticker and m_a.series_ticker == m_b.series_ticker
            and shared_tags):
        if m_a.end_date_iso < m_b.end_date_iso:
            return "a_outer_b_inner"
        elif m_b.end_date_iso < m_a.end_date_iso:
            return "b_outer_a_inner"

    return None


def _extract_threshold(question: str) -> Optional[float]:
    """
    Extract dollar threshold from question text.
    Handles: "BTC above $100,000", "ETH > 4000 USD", "S&P 500 closes above 5000"
    """
    import re
    patterns = [
        r'\$([0-9,]+(?:\.[0-9]+)?)',
        r'>(?: greater than )?([0-9,]+(?:\.[0-9]+)?)\s*(?:USD|USDC)?',
        r'above\s+([0-9,]+(?:\.[0-9]+)?)',
    ]
    for pattern in patterns:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            num_str = m.group(1).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                pass
    return None


def _normalize_price(price: float) -> float:
    """Normalize to 0-1 range: Kalshi is in cents (0-100), Polymarket is 0-1."""
    if price > 1:
        return price / 100.0
    return price


def _constraint_confidence(outer: Market, inner: Market) -> float:
    """
    Score how confident we are that this is a valid constraint.
    - Same condition_id (Polymarket) = very high confidence
    - Same series_ticker + similar tags = high confidence
    - Shared keywords only = medium
    """
    if outer.condition_id and outer.condition_id == inner.condition_id:
        return 0.95
    if outer.series_ticker and outer.series_ticker == inner.series_ticker:
        return 0.85
    shared = set(outer.tags) & set(inner.tags)
    if shared:
        return 0.60 + 0.1 * min(len(shared), 3)
    return 0.40


def _deduplicate_opportunities(opps: list[CombinatorialOpportunity]) -> list[CombinatorialOpportunity]:
    """Remove duplicate (A,B)/(B,A) pairs, keep larger edge."""
    seen: dict[str, CombinatorialOpportunity] = {}
    for opp in opps:
        key = frozenset({opp.outer_market.id, opp.inner_market.id})
        if key not in seen or opp.edge_per_dollar > seen[key].edge_per_dollar:
            seen[key] = opp
    return list(seen.values())


# ---------------------------------------------------------------------------
# Step 3: Full scan function
# ---------------------------------------------------------------------------

def scan_for_combinatorial_opportunities(
    markets: list[Market],
    min_edge_per_dollar: float = 0.01,
    max_days_apart: int = 30,
) -> list[CombinatorialOpportunity]:
    """
    Main entry point: scan a list of markets for combinatorial arb.
    
    Args:
        markets: List of Market objects from Kalshi and Polymarket
        min_edge_per_dollar: Minimum profit per $1 to report
        max_days_apart: Max days between market end_dates to consider
    
    Returns:
        List of CombinatorialOpportunity objects sorted by edge descending
    """
    # 1. Cluster markets by shared underlying
    clusters = build_market_clusters(markets, max_days_apart)
    
    # 2. Check combinatorial constraints within each cluster
    all_opportunities = []
    for cluster_key, cluster_markets in clusters.items():
        if len(cluster_markets) < 2:
            continue
        opps = check_combinatorial_constraints(cluster_markets)
        all_opportunities.extend(opps)
    
    # 3. Filter by minimum edge threshold
    filtered = [o for o in all_opportunities if o.edge_per_dollar >= min_edge_per_dollar]
    
    # 4. Sort by edge descending
    filtered.sort(key=lambda o: o.edge_per_dollar, reverse=True)
    
    return filtered


# ---------------------------------------------------------------------------
# Step 4: Format for human review
# ---------------------------------------------------------------------------

def format_opportunity(opp: CombinatorialOpportunity) -> str:
    """Format a CombinatorialOpportunity as a human-readable alert."""
    outer = opp.outer_market
    inner = opp.inner_market
    edge_pct = opp.edge_per_dollar * 100
    
    return f"""
⚠️  COMBINATORIAL ARB — {opp.direction.replace("_", " ").upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Outer (buy):   {outer.question[:80]}
  Platform:    {outer.platform} | Price: ${outer.yes_ask:.4f} YES
  End date:    {outer.end_date_iso[:10]}

Inner (sell):  {inner.question[:80]}
  Platform:    {inner.platform} | Price: ${inner.yes_ask:.4f} YES
  End date:    {inner.end_date_iso[:10]}

Edge:          {edge_pct:.2f}% per dollar
Ratio:         P(inner)/P(outer) = {opp.implied_prob_ratio:.4f}
Confidence:    {opp.confidence:.0%}
""".strip()


# ---------------------------------------------------------------------------
# Demo / test
# ---------------------------------------------------------------------------

def _demo():
    """Build synthetic markets and demonstrate the algorithm."""
    
    markets = [
        # BTC above $100k by end of year — outer (higher threshold)
        Market(
            id="kalshi_btc_100k",
            question="Will BTC be above $100,000 by December 31 2026?",
            platform="kalshi", yes_bid=0.68, yes_ask=0.70,
            no_bid=0.28, no_ask=0.30,
            end_date_iso="2026-12-31T23:59:00Z",
            market_id="btc_100k_2026", series_ticker="BTC.USDT.USDC",
            tags=["btc", "price", "above", "yearend"],
            condition_id=""
        ),
        # BTC above $110k by end of year — inner (higher threshold implies outer)
        Market(
            id="kalshi_btc_110k",
            question="Will BTC be above $110,000 by December 31 2026?",
            platform="kalshi", yes_bid=0.58, yes_ask=0.60,
            no_bid=0.38, no_ask=0.40,
            end_date_iso="2026-12-31T23:59:00Z",
            market_id="btc_110k_2026", series_ticker="BTC.USDT.USDC",
            tags=["btc", "price", "above", "yearend"],
            condition_id=""
        ),
        # Polymarket version of same — should also trigger
        Market(
            id="poly_btc_100k",
            question="Will BTC be above $100,000 by December 31 2026?",
            platform="polymarket", yes_bid=0.69, yes_ask=0.71,
            no_bid=0.27, no_ask=0.29,
            end_date_iso="2026-12-31T23:59:00Z",
            market_id="", series_ticker="BTC.USDT.USDC",
            tags=["btc", "price", "above", "yearend"],
            condition_id="cond_btc_100k"
        ),
        # Non-related market — should NOT trigger
        Market(
            id="kalshi_nba",
            question="Will the Lakers win the 2026 NBA Finals?",
            platform="kalshi", yes_bid=0.20, yes_ask=0.22,
            no_bid=0.76, no_ask=0.78,
            end_date_iso="2026-06-30T23:59:00Z",
            market_id="nba_finals_2026", series_ticker="NBA.2026",
            tags=["nba", "basketball", "lakers", "finals"],
            condition_id=""
        ),
    ]
    
    print("=" * 60)
    print("COMBINATORIAL ARBITRAGE SCAN — DEMO")
    print("=" * 60)
    
    opps = scan_for_combinatorial_opportunities(markets, min_edge_per_dollar=0.01)
    
    if not opps:
        print("\nNo opportunities found.")
    else:
        for opp in opps:
            print(format_opportunity(opp))
            print()


if __name__ == "__main__":
    _demo()
