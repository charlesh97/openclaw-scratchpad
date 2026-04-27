# Combinatorial Arbitrage

**Status:** ✅ YES — High Confidence  
**Source:** [arXiv:2508.03474 — "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"](https://arxiv.org/abs/2508.03474)  
**Raw PDF:** <https://arxiv.org/pdf/2508.03474>  
**Authors:** Saguillo, Ghafouri, Kiffer, Suarez-Tangil (2025)  
**Research date:** 2026-04-26 by vega

---

## What It Is

Traditional arbitrage in prediction markets exploits the fact that for any binary contract: `YES + NO = $1` (minus spread). If the sum is under $1, you lock in a guaranteed profit.

**Combinatorial Arbitrage** exploits a more subtle constraint: when one outcome *logically implies* another outcome in a separate market, prices must respect that logical relationship — and they often don't.

The paper identifies **$40M+ of profit extracted** on Polymarket from this strategy, confirming it's real, monetized edge that arb-bot-main does not currently detect.

---

## The Core Intuition

### Threshold Nesting
For any underlying asset, if market A = "BTC above $X by date T" and market B = "BTC above $Y by date T" where Y > X, then:

> **B implies A** (if BTC closes above $110k, it must also close above $100k)  
> Therefore: `P(B) ≤ P(A)` must hold (monotonicity constraint)

If `P(B) > P(A)` — the inner/more-specific contract is priced *higher* than the outer/more-general one — you have a violation. The trade:

- **Buy B** (the implied/inner market, priced high)  
- **Sell A** (the implying/outer market, priced low)  
- At resolution: if BTC > $110k → B pays $1, A pays $1 → net = $0 but you collected the price difference upfront ✓  
- At resolution: if $100k < BTC ≤ $110k → B loses $1, A pays $1 → net = $0 ✓  
- At resolution: if BTC ≤ $100k → both lose $1 → net = -$1 (bounded loss, not unbounded)

The edge is: `P(B) - P(A)` collected at entry. It's guaranteed because the two contracts' payouts structurally overlap.

### Temporal Nesting
Same logic applies to different expiry dates on the same underlying. If A = "BTC above $100k by Dec 31" and B = "BTC above $100k by Nov 30", then B implies A (B resolves earlier). `P(B) ≥ P(A)` must hold. Violations are rarer but exist.

---

## Why $40M+ Was Extracted

The paper documents that combinatorial violations are persistent because:
1. **Platform separation** — Kalshi and Polymarket have different liquidity pools, different traders, different information processing speeds
2. **End-date inconsistencies** — Polymarket markets with the same `market_id` can have different `end_date_iso` values, creating hidden logical relationships
3. **O(2^n) detection problem** — naive checking of all market pairs is computationally intractable; heuristic reduction is essential — which means most violations go undetected by naive scanners
4. **Human traders can't process at scale** — the relationships are too many and too complex without systematic detection

---

## Architecture

```
arb-bot-main/
├── algos/
│   ├── common/
│   │   ├── base_bot.py       ← Opportunity dataclass lives here
│   │   └── opportunity.py
│   ├── parity/               ← existing
│   ├── temporal/             ← existing
│   └── combinatorial/         ← NEW: this module
│       ├── __init__.py
│       ├── cluster.py         ← market clustering logic
│       ├── constraints.py     ← combinatorial constraint checking
│       └── scanner.py         ← full scan entrypoint
```

**Integration point:** `scan_for_combinatorial_opportunities()` returns a list of `CombinatorialOpportunity` objects (subclass of `Opportunity`). These slot into the existing pipeline alongside `check_parity()` and `check_temporal()` results.

**Key design decisions:**
- Market clustering is done first to reduce O(n²) pair checks to O(n × k) where k = avg cluster size
- Clustering uses `series_ticker` as primary key + keyword tags as fallback
- Only same-month clusters are checked (further reduces false pairs)
- Threshold extraction via regex from question text (no LLM needed)

---

## Sample Code

```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class Market:
    id: str
    question: str
    platform: Literal["kalshi", "polymarket"]
    yes_ask: float          # best ask (we buy at ask)
    no_ask: float
    end_date_iso: str       # ISO timestamp
    series_ticker: str      # e.g. "BTC.USDT.USDC"
    tags: list[str]
    condition_id: str       # Polymarket condition_id


@dataclass
class CombinatorialOpportunity:
    outer_market: Market    # the more-general market
    inner_market: Market   # the more-specific market (implies outer)
    direction: str          # "buy_outer_sell_inner"
    edge_per_dollar: float
    confidence: float      # 0-1


def check_combinatorial_constraints(markets: list[Market]) -> list[CombinatorialOpportunity]:
    """
    For a cluster of same-asset, same-month markets,
    find pairs where P(inner) > P(outer) — a guaranteed arb.
    """
    opps = []
    for i, m_a in enumerate(markets):
        for m_b in markets[i+1:]:
            opp = _evaluate_pair(m_a, m_b)
            if opp is not None and opp.edge_per_dollar > 0.005:
                opps.append(opp)
    return opps


def _evaluate_pair(m_a: Market, m_b: Market) -> Optional[CombinatorialOpportunity]:
    """
    Detect nesting direction and check for price violation.
    Returns CombinatorialOpportunity if P(inner) > P(outer).
    """
    direction = _detect_nesting_direction(m_a, m_b)
    if direction is None:
        return None

    outer, inner = (m_a, m_b) if direction == "a_outer_b_inner" else (m_b, m_a)
    p_outer = m2u(outer.yes_ask)
    p_inner = m2u(inner.yes_ask)

    if p_inner > p_outer:
        return CombinatorialOpportunity(
            outer_market=outer,
            inner_market=inner,
            direction="buy_outer_sell_inner",
            edge_per_dollar=p_inner - p_outer,
            confidence=_constraint_confidence(outer, inner),
        )
    return None


def _detect_nesting_direction(m_a: Market, m_b: Market) -> Optional[str]:
    """Returns 'a_outer_b_inner' if A implies B, 'b_outer_a_inner' if B implies A."""
    thresh_a = _extract_threshold(m_a.question)
    thresh_b = _extract_threshold(m_b.question)

    if thresh_a is not None and thresh_b is not None:
        if thresh_a > thresh_b:
            return "b_outer_a_inner"   # higher threshold = inner (more specific)
        elif thresh_b > thresh_a:
            return "a_outer_b_inner"

    # Temporal nesting
    shared = set(m_a.tags) & set(m_b.tags)
    if shared and m_a.end_date_iso != m_b.end_date_iso:
        return "a_outer_b_inner" if m_a.end_date_iso < m_b.end_date_iso else "b_outer_a_inner"

    return None


def _extract_threshold(question: str) -> Optional[float]:
    """Extract dollar threshold from question text."""
    import re
    m = re.search(r'\$([0-9,]+(?:\.[0-9]+)?)', question, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def m2u(price: float) -> float:
    """Normalize: Kalshi is cents (0-100), Polymarket is 0-1."""
    return price / 100.0 if price > 1 else price
```

---

## Key Risks and Guard Rails

| Risk | Mitigation |
|---|---|
| False nesting detection (keyword confusion) | Confidence threshold — only trade at `confidence ≥ 0.75` |
| Platform-level execution risk | Only trade when both legs can be filled simultaneously or within 60s window |
| End-date mismatch (same event, different platform times) | Use `end_date_iso` comparison with ±1 day tolerance |
| Thin liquidity (can't close both legs) | Minimum daily volume filter per market |
| Regime shift (arb presence increases, edge disappears) | Monitor edge magnitude — if avg edge drops below 0.5%, alert + pause |

---

## Next Steps

1. **Build market cluster index** keyed by `series_ticker + YYYY-MM` from end_date
2. **Integrate into arb-bot scan loop** alongside existing `check_*` functions
3. **Add confidence gate** — only surface opportunities with `confidence ≥ 0.75`
4. **Backtest** on 90 days of historical market data using Kalshi/Polymarket historical APIs
5. **Paper link in arb-bot README:** <https://arxiv.org/abs/2508.03474>
