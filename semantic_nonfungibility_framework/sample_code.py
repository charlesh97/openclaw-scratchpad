"""
Semantic Non-Fungibility Framework — Cross-Platform Event Identity Resolution

Based on arXiv:2601.01706 (Gebele & Matthes, TUM, January 2026)
Dataset: 100,000+ events across 10 prediction market venues (2018–2025)

Key findings:
- ~6% of events listed on multiple platforms simultaneously
- Average execution-aware price deviation: 2–4% for semantically equivalent pairs
- Structural friction (not info disagreement) drives persistent mispricings

This prototype:
1. Encodes event descriptions using sentence embeddings
2. Computes semantic similarity scores
3. Flags cross-platform pairs with price deviations exceeding transaction costs
4. Estimates capital-at-risk profit with lock-up duration adjustment
"""

import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class Market:
    venue: str           # e.g., "Polymarket", "Kalshi"
    event_id: str
    description: str
    resolution_rule: str
    cutoff: str          # ISO timestamp of market close
    yes_price: float
    no_price: float
    shares_liquidity: float = 0.0

@dataclass
class CrossPlatformPair:
    market_a: Market
    market_b: Market
    semantic_similarity: float   # 0–1 (embedding cosine similarity)
    price_deviation: float       # abs(price_a - price_b), e.g., 0.03 = 3%
    effective_edge: float        # price_deviation minus fees
    estimated_lockup_hours: float
    capital_at_risk: float       # $ needed to execute one side
    annualized_return_pct: float
    is_semantically_verified: bool  # similarity above threshold
    risk_flags: list[str]

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Dot-product cosine similarity between two embedding vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a**2 for a in vec_a))
    norm_b = math.sqrt(sum(b**2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def embedding_similarity(desc_a: str, desc_b: str) -> float:
    """
    Placeholder: Replace with sentence-transformers or OpenAI embedding API.
    Returns cosine similarity between 0 and 1.

    Example (pseudo):
        emb_a = embed(desc_a)
        emb_b = embed(desc_b)
        return cosine_similarity(emb_a, emb_b)
    """
    # Naive word-overlap baseline — swap for real embeddings in production
    words_a = set(desc_a.lower().split())
    words_b = set(desc_b.lower().split())
    intersection = words_a & words_b
    union = words_a | words_b
    if not union:
        return 0.0
    return len(intersection) / len(union)

def check_cross_platform_arb(market_a: Market,
                             market_b: Market,
                             taker_fee_pct: float = 0.0175,
                             lockup_hours: float = 24.0,
                             min_similarity: float = 0.75) -> CrossPlatformPair:
    """
    Evaluate a cross-platform opportunity between two markets.

    Args:
        taker_fee_pct: Combined fee both legs (default 1.75% per side)
        lockup_hours: How long capital is locked until market resolution
        min_similarity: Semantic similarity threshold for "same event" confidence
    """
    sim = embedding_similarity(market_a.description, market_b.description)
    # Also compare resolution rules for stricter matching
    same_resolution = market_a.resolution_rule.strip().lower() == market_b.resolution_rule.strip().lower()
    same_cutoff = market_a.cutoff == market_b.cutoff

    # Price deviation: use YES price on side A vs NO price on side B
    # (or whichever combination gives the best cross-platform edge)
    p_a = market_a.yes_price
    p_b = market_b.yes_price
    deviation = abs(p_a - p_b)

    # Effective edge after taker fees on BOTH legs
    total_fees = 2 * taker_fee_pct
    effective_edge = deviation - total_fees

    # Capital at risk: hold the more expensive leg until resolution
    capital_needed = max(p_a, p_b)

    # Annualized return (assuming repeated deployment)
    if lockup_hours > 0 and capital_needed > 0:
        annualized = (effective_edge / capital_needed) * (365 * 24 / lockup_hours) * 100
    else:
        annualized = 0.0

    risk_flags = []
    if sim < min_similarity:
        risk_flags.append("LOW_SEMANTIC_SIMILARITY")
    if not same_resolution:
        risk_flags.append("RESOLUTION_RULES_DIFFER")
    if not same_cutoff:
        risk_flags.append("CUTOFF_TIMES_DIFFER")
    if effective_edge <= 0:
        risk_flags.append("EDGE_NEGATIVE_AFTER_FEES")
    if lockup_hours > 72:
        risk_flags.append("HIGH_LOCKUP_RISK")

    return CrossPlatformPair(
        market_a=market_a,
        market_b=market_b,
        semantic_similarity=round(sim, 3),
        price_deviation=round(deviation, 4),
        effective_edge=round(effective_edge, 4),
        estimated_lockup_hours=lockup_hours,
        capital_at_risk=round(capital_needed, 4),
        annualized_return_pct=round(annualized, 2),
        is_semantically_verified=(sim >= min_similarity and same_resolution),
        risk_flags=risk_flags
    )

def format_opportunity(pair: CrossPlatformPair) -> str:
    """Human-readable format for an opportunity."""
    status = "✅ VERIFIED" if pair.is_semantically_verified else "⚠️ UNVERIFIED"
    edge_str = f"{pair.effective_edge*100:.2f}%"
    ann_str = f"{pair.annualized_return_pct:.1f}%"
    flags = "; ".join(pair.risk_flags) if pair.risk_flags else "none"

    return f"""[{pair.market_a.venue} ↔ {pair.market_b.venue}] {status}
  Events: "{pair.market_a.description[:60]}" ↔ "{pair.market_b.description[:60]}"
  Similarity: {pair.semantic_similarity:.2f}  |  Deviation: {pair.price_deviation*100:.2f}%
  Edge (after fees): {edge_str}  |  Annualized: {ann_str}
  Capital at risk: ${pair.capital_at_risk:.4f}  |  Lock-up: {pair.estimated_lockup_hours:.0f}h
  Risk flags: {flags}
"""

# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulated Polymarket vs Kalshi aligned market pair
    # "Will BTC close above $100k by end of March 2026?"
    poly_market = Market(
        venue="Polymarket",
        event_id="btc_march2026",
        description="Will Bitcoin close above $100,000 by March 31, 2026?",
        resolution_rule="BTC/USD price from CoinDesk closing price on March 31, 2026",
        cutoff="2026-03-31T23:59:59Z",
        yes_price=0.45,
        no_price=0.55,
        shares_liquidity=200.0
    )

    kalshi_market = Market(
        venue="Kalshi",
        event_id="btc_q1_2026",
        description="Will Bitcoin be above $100,000 at the end of Q1 2026?",
        resolution_rule="BTC/USD price from CoinDesk at March 31, 2026 11:59pm UTC",
        cutoff="2026-03-31T23:59:59Z",
        yes_price=0.51,
        no_price=0.49,
        shares_liquidity=150.0
    )

    result = check_cross_platform_arb(
        poly_market, kalshi_market,
        taker_fee_pct=0.0175,
        lockup_hours=48.0
    )
    print(format_opportunity(result))
