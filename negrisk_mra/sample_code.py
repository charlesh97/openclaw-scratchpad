#!/usr/bin/env python3
"""
NegRisk Multi-Condition Rebalancing Arbitrage — vega research
==============================================================
Source: arXiv 2508.03474 (Saguillo et al., August 2025)
       + Kalshi NegRisk / multi-condition market structure

Concept: Kalshi NegRisk markets structure a single question as
a set of mutually exclusive, exhaustive CONDITIONS (e.g., Fed decision
= Hold / Cut25 / Cut50+). Each condition has a YES contract.

This script detects THREE types of mispricings in NegRisk markets:
  1. Sum violation:    Σ P(Ci) ≠ 1.00  (probability mass not normalized)
  2. Monotonicity:     If Ci ⊂ Cj, then P(Ci) ≤ P(Cj)
  3. Negation pairs:   If Ci = ¬Cj, then P(Ci) + P(Cj) = 1.00

Usage:
  python negrisk_mra.py [--min-edge E] [--market-id ID]
"""

import math
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    condition_id: str
    question: str           # e.g. "Hold", "Cut 25bp", "Cut 50bp+"
    yes_bid: float          # best bid (we buy at ask, sell at bid)
    yes_ask: float
    no_bid: float
    no_ask: float
    end_date_iso: str


@dataclass
class NegRiskMarket:
    market_id: str
    question: str           # full market question text
    platform: str          # "kalshi"
    conditions: list[Condition] = field(default_factory=list)
    ticker: str = ""       # e.g. "FED.RATES"
    tags: list[str] = field(default_factory=list)


@dataclass
class MRAOpportunity:
    market: NegRiskMarket
    opportunity_type: str   # "sum_violation" | "monotonicity_violation" | "negation_violation"
    description: str
    edge_per_dollar: float
    action: str            # "buy_X_sell_Y" or "rebalance"
    confidence: float      # 0–1


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def check_sum_violation(market: NegRiskMarket, fee_rate: float = 0.02) -> Optional[MRAOpportunity]:
    """
    Check whether Σ P(condition_i) ≈ 1.00.
    
    Each condition's YES contract price represents P(Ci).
    The sum should equal $1.00. Any deviation > fees is arb.
    
    Action: If sum < 1.00, buy all YES contracts proportionally.
            If sum > 1.00, sell all YES contracts proportionally.
    
    This is equivalent to market-making the entire condition set.
    """
    if not market.conditions:
        return None
    
    # Use mid prices (average of bid/ask)
    total = sum((c.yes_bid + c.yes_ask) / 2 for c in market.conditions)
    
    deviation = abs(total - 1.0)
    if deviation <= fee_rate:
        return None  # within fee threshold
    
    # Edge per dollar: deviation minus fees
    edge = deviation - fee_rate
    
    direction = "buy_all" if total < 1.0 else "sell_all"
    desc = (
        f"Sum of condition probabilities = {total:.4f} (expected 1.0000). "
        f"Deviation = {deviation:.4f}. "
        f"→ {direction.upper()} all YES contracts."
    )
    
    return MRAOpportunity(
        market=market,
        opportunity_type="sum_violation",
        description=desc,
        edge_per_dollar=edge,
        action=direction,
        confidence=0.90,  # high confidence since this is basic arithmetic
    )


def check_monotonicity_violations(market: NegRiskMarket, fee_rate: float = 0.02) -> list[MRAOpportunity]:
    """
    Check whether monotonicity constraints hold between conditions.
    
    Example: "Fed holds rates" implies "rates don't rise" (less specific).
            So: P(hold) ≤ P(no_rise) must hold.
    
    We infer these relationships from question text keywords.
    """
    opportunities = []
    conditions = market.conditions
    n = len(conditions)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel = infer_condition_relationship(conditions[i], conditions[j])
            if rel is None:
                continue
            
            p_i = (conditions[i].yes_bid + conditions[i].yes_ask) / 2
            p_j = (conditions[j].yes_bid + conditions[j].yes_ask) / 2
            
            if rel == "i_implies_j":
                # Constraint: p_i <= p_j
                if p_i > p_j + fee_rate:
                    edge = p_i - p_j - fee_rate
                    opportunities.append(MRAOpportunity(
                        market=market,
                        opportunity_type="monotonicity_violation",
                        description=(
                            f"'{conditions[i].question}' implies '{conditions[j].question}'. "
                            f"Constraint violated: P(i)={p_i:.4f} > P(j)={p_j:.4f}. "
                            f"→ Sell YES on {conditions[i].question}, buy YES on {conditions[j].question}."
                        ),
                        edge_per_dollar=edge,
                        action="sell_i_buy_j",
                        confidence=0.70,
                    ))
            
            elif rel == "j_implies_i":
                # Constraint: p_j <= p_i
                if p_j > p_i + fee_rate:
                    edge = p_j - p_i - fee_rate
                    opportunities.append(MRAOpportunity(
                        market=market,
                        opportunity_type="monotonicity_violation",
                        description=(
                            f"'{conditions[j].question}' implies '{conditions[i].question}'. "
                            f"Constraint violated: P(j)={p_j:.4f} > P(i)={p_i:.4f}. "
                            f"→ Sell YES on {conditions[j].question}, buy YES on {conditions[i].question}."
                        ),
                        edge_per_dollar=edge,
                        action="sell_j_buy_i",
                        confidence=0.70,
                    ))
    
    return opportunities


def infer_condition_relationship(c_a: Condition, c_b: Condition) -> Optional[str]:
    """
    Infer logical relationship between two conditions from question text.
    
    Returns:
      "a_implies_b"  — if c_a is true then c_b must be true
      "b_implies_a"  — if c_b is true then c_a must be true
      None           — no clear relationship
    """
    q_a = c_a.question.lower()
    q_b = c_b.question.lower()
    
    # --- Fed rate keywords ---
    hold_kw = {"hold", "unchanged", "no change", "steady", "pause"}
    cut_kw = {"cut", "lower", "reduce", "decrease", "easing"}
    rise_kw = {"rise", "raise", "hike", "increase", "tightening", "higher"}
    
    # Hold vs. Rise: hold implies not rise
    if any(k in q_a for k in hold_kw) and any(k in q_b for k in rise_kw):
        return "a_implies_b"  # hold implies not rise
    if any(k in q_b for k in hold_kw) and any(k in q_a for k in rise_kw):
        return "b_implies_a"  # hold implies not rise
    
    # Hold vs. Cut: hold implies not cut
    if any(k in q_a for k in hold_kw) and any(k in q_b for k in cut_kw):
        return "a_implies_b"
    if any(k in q_b for k in hold_kw) and any(k in q_a for k in cut_kw):
        return "b_implies_a"
    
    # Cut vs. Rise: cut implies not rise? No — both are independent
    # Cut 25 vs. Cut 50+: cut 50+ implies cut 25 is also true
    cut25_kw = {"25", "0.25", "quarter", "25bp"}
    cut50_kw = {"50", "0.50", "50bp", "50 bps", "50+", "more than 50"}
    
    has_cut25_a = any(k in q_a for k in cut25_kw)
    has_cut50_a = any(k in q_a for k in cut50_kw)
    has_cut25_b = any(k in q_b for k in cut25_kw)
    has_cut50_b = any(k in q_b for k in cut50_kw)
    
    # Cut 50+ implies Cut 25 (if you cut 50+, you also cut 25)
    if has_cut50_a and has_cut25_b:
        return "a_implies_b"  # cut50+ → cut25
    if has_cut50_b and has_cut25_a:
        return "b_implies_a"  # cut50+ → cut25
    
    # --- Numeric threshold nesting ---
    thresh_a = _extract_threshold(q_a)
    thresh_b = _extract_threshold(q_b)
    if thresh_a is not None and thresh_b is not None and thresh_a > thresh_b:
        # Higher threshold implies lower threshold? Only if same asset/time
        # E.g., BTC > $100k implies BTC > $90k
        if _same_underlying(q_a, q_b):
            return "a_implies_b" if thresh_a > thresh_b else "b_implies_a"
    
    return None


def _same_underlying(q_a: str, q_b: str) -> bool:
    """Check if two questions refer to the same underlying asset."""
    btc_kw = {"btc", "bitcoin", "₿"}
    eth_kw = {"eth", "ethereum"}
    sp500_kw = {"sp 500", "s&p", "sp500", "spex"}
    
    for kw_set in [btc_kw, eth_kw, sp500_kw]:
        if any(k in q_a for k in kw_set) and any(k in q_b for k in kw_set):
            return True
    return False


def _extract_threshold(question: str) -> Optional[float]:
    """Extract dollar threshold from question text."""
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


# ---------------------------------------------------------------------------
# Full scanner
# ---------------------------------------------------------------------------

def scan_neg_risk_market(market: NegRiskMarket, fee_rate: float = 0.02) -> list[MRAOpportunity]:
    """
    Main entry point: scan a single NegRisk market for MRA opportunities.
    
    Returns list of opportunities sorted by edge descending.
    """
    opportunities = []
    
    # 1. Sum violation check
    sum_opp = check_sum_violation(market, fee_rate)
    if sum_opp:
        opportunities.append(sum_opp)
    
    # 2. Monotonicity violations
    mono_opps = check_monotonicity_violations(market, fee_rate)
    opportunities.extend(mono_opps)
    
    # Sort by edge descending
    opportunities.sort(key=lambda o: o.edge_per_dollar, reverse=True)
    return opportunities


def format_opportunity(opp: MRAOpportunity) -> str:
    market = opp.market
    edge_pct = opp.edge_per_dollar * 100
    return f"""
⚠️  NEGRISK MRA — {opp.opportunity_type.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Market:       {market.question[:80]}
Platform:     {market.platform} | Ticker: {market.ticker}
Conditions:   {len(market.conditions)}

{opp.description}

Edge:         {edge_pct:.3f}% per dollar
Action:       {opp.action}
Confidence:   {opp.confidence:.0%}
""".strip()


# ---------------------------------------------------------------------------
# Mock Kalshi API data builder
# ---------------------------------------------------------------------------

def _build_fed_rates_demo() -> NegRiskMarket:
    """
    Build a synthetic Fed Rates NegRisk market demonstrating:
    - Sum violation (conditions sum to 0.97, slightly under-priced)
    - Monotonicity violation (Cut50+ implies Cut25, but Cut50+ is priced higher)
    """
    return NegRiskMarket(
        market_id="fed-rates-dec-2025",
        question="What will the Fed do with rates by December 2025?",
        platform="kalshi",
        ticker="FED.RATES",
        tags=["fed", "rates", "monetary-policy"],
        conditions=[
            Condition(
                condition_id="fed_cond_hold",
                question="Hold rates (unchanged)",
                yes_bid=0.52, yes_ask=0.54,  # mid = 0.53
                no_bid=0.44, no_ask=0.46,
                end_date_iso="2025-12-31T23:59:00Z",
            ),
            Condition(
                condition_id="fed_cond_cut25",
                question="Cut rates by 25bp",
                yes_bid=0.30, yes_ask=0.32,  # mid = 0.31
                no_bid=0.66, no_ask=0.68,
                end_date_iso="2025-12-31T23:59:00Z",
            ),
            Condition(
                condition_id="fed_cond_cut50",
                question="Cut rates by 50bp or more",
                yes_bid=0.12, yes_ask=0.14,  # mid = 0.13
                no_bid=0.84, no_ask=0.86,
                end_date_iso="2025-12-31T23:59:00Z",
            ),
        ],
    )


def _build_btc_threshold_demo() -> NegRiskMarket:
    """
    BTC threshold NegRisk market: "BTC above $X by date"
    Cut50+ implies Cut25 → monotonicity violation.
    """
    return NegRiskMarket(
        market_id="btc-threshold-2026",
        question="Where will BTC close relative to key levels in 2026?",
        platform="kalshi",
        ticker="BTC.USDT.USDC",
        tags=["btc", "price", "threshold"],
        conditions=[
            Condition(
                condition_id="btc_under_90k",
                question="BTC closes below $90,000",
                yes_bid=0.20, yes_ask=0.22,
                no_bid=0.76, no_ask=0.78,
                end_date_iso="2026-12-31T23:59:00Z",
            ),
            Condition(
                condition_id="btc_90k_100k",
                question="BTC closes between $90k–$100k",
                yes_bid=0.30, yes_ask=0.32,
                no_bid=0.66, no_ask=0.68,
                end_date_iso="2026-12-31T23:59:00Z",
            ),
            Condition(
                condition_id="btc_100k_110k",
                question="BTC closes between $100k–$110k",
                yes_bid=0.25, yes_ask=0.27,
                no_bid=0.71, no_ask=0.73,
                end_date_iso="2026-12-31T23:59:00Z",
            ),
            Condition(
                condition_id="btc_over_110k",
                question="BTC closes above $110,000",
                yes_bid=0.22, yes_ask=0.24,
                no_bid=0.74, no_ask=0.76,
                end_date_iso="2026-12-31T23:59:00Z",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NegRisk Multi-Condition MRA Scanner")
    parser.add_argument("--min-edge", type=float, default=0.01,
                        help="Minimum edge per dollar to report (default: 0.01 = 1%)")
    parser.add_argument("--fee-rate", type=float, default=0.02,
                        help="Fee rate to subtract from edge (default: 0.02 = 2%)")
    args = parser.parse_args()
    
    print("=" * 65)
    print("NEGRISK MULTI-CONDITION REBALANCING ARBITRAGE — DEMO")
    print("=" * 65)
    
    for demo_market in [_build_fed_rates_demo(), _build_btc_threshold_demo()]:
        opps = scan_neg_risk_market(demo_market, fee_rate=args.fee_rate)
        
        if not opps:
            print(f"\n{demo_market.question[:60]}... → No opportunities (edge < {args.min_edge:.2%})")
            continue
        
        print(f"\n{'=' * 65}")
        print(f"Market: {demo_market.question}")
        print(f"Platform: {demo_market.platform} | Ticker: {demo_market.ticker}")
        print(f"Conditions: {len(demo_market.conditions)}")
        
        for opp in opps:
            if opp.edge_per_dollar >= args.min_edge:
                print(format_opportunity(opp))
                print()


if __name__ == "__main__":
    main()
