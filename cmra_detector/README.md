# Cross-Market Rebalancing Arbitrage (CMRA)

**Status:** YES — High Confidence  
**Source:** Saguillo et al., *arXiv:2508.03474* (August 2025)  
**Research date:** 2026-04-27 by vega

---

## TL;DR

CMRA exploits mispricings between **two separate markets** that share a condition — e.g. "BTC > $100k by Dec 31" and "BTC > $110k by Dec 31". When their prices violate the monotonicity constraint (higher threshold should have lower probability), you buy the overpriced leg and sell the underpriced leg. Accounts for ~99.76% of the $40M extracted on Polymarket. Much more prevalent than nested combinatorial arb.

---

## The Core Insight

For any set of threshold markets on the same underlying:

```
P(BTC > $100k) ≥ P(BTC > $110k) ≥ P(BTC > $120k)
```

The probabilities must be weakly decreasing as thresholds increase. When the market violates this — priced markets imply a higher probability for the higher threshold — you have a CMRA signal.

**Example:**
- Market A (Kalshi): "BTC > $100k by Dec 31" → YES = 80¢
- Market B (Polymarket): "BTC > $105k by Dec 31" → YES = 85¢ ← VIOLATION

85¢ > 80¢ means traders are pricing the higher threshold as *more likely*, which is structurally impossible. The arb: sell YES(B) + buy YES(A), capturing the mispricing.

---

## How It Differs from Standard Parity Arb

| | Parity Arb (MRA) | CMRA |
|---|---|---|
| Markets | YES + NO on same contract | Two *separate* markets, same underlying |
| Condition | Same market | Share a condition or semantic topic |
| Edge source | YES+NO sums below $1 | Price ordering violation across thresholds |
| Prevalence | Common | Very common (~99.76% of $40M extracted) |

---

## Architecture

```
cmra_detector.py
├── Market          dataclass — market contract with threshold + price
├── CMRAOpportunity dataclass — detected signal with spread, size, direction
└── CMRADetector    class
    ├── check_monotonicity()   — scan a group of markets for ordering violations
    └── scan_universe()       — group by underlying, scan all groups
```

---

## Sample Code (key detection logic)

```python
from dataclasses import dataclass

@dataclass
class Market:
    market_id: str
    platform: str          # "polymarket" or "kalshi"
    title: str
    underlying: str        # "BTC", "ETH", "SPX"
    threshold: float        # price threshold in dollars
    yes_price: float       # current YES price ($0.00–$1.00)
    liquidity: float       # available volume

def detect_cmra(markets: list[Market], min_spread: float = 0.005) -> list[dict]:
    """
    Scan threshold-ordered markets for CMRA violations.
    A violation occurs when P(higher_threshold) > P(lower_threshold).

    Strategy: buy the underpriced lower-threshold YES +
              sell the overpriced higher-threshold YES
    Net spread = (lower.yes_price - higher.yes_price) adjusted for fees
    """
    opportunities = []
    # Sort by threshold ascending (lowest threshold first)
    sorted_markets = sorted(
        [m for m in markets if m.threshold is not None],
        key=lambda m: m.threshold
    )

    for i in range(len(sorted_markets) - 1):
        lower = sorted_markets[i]
        higher = sorted_markets[i + 1]

        # VIOLATION: higher threshold priced higher = structurally impossible
        if higher.yes_price > lower.yes_price:
            gross_spread = higher.yes_price - lower.yes_price
            net_spread = gross_spread - 0.02  # ~2¢ fees

            if net_spread >= min_spread:
                opportunities.append({
                    "lower_market": lower.market_id,
                    "higher_market": higher.market_id,
                    "lower_threshold": lower.threshold,
                    "higher_threshold": higher.threshold,
                    "gross_spread": gross_spread,
                    "net_spread": net_spread,
                    "size_limit": min(lower.liquidity, higher.liquidity),
                    "direction": "buy_lower_sell_higher",
                    "invalidation": "market moves, liquidity insufficient, fee underestimate"
                })

    return opportunities
```

---

## Key Risks

| Risk | Mitigation |
|---|---|
| Threshold grouping errors | Require shared condition_id or high embedding similarity |
| Liquidity at execution | Size limit = min(both legs); execute below limit |
| Fees eat the edge | Net spread must exceed ~2¢ before considering |
| Short-duration markets | CMRA works on any duration; shorter = more frequent checks needed |

---

## Next Steps

1. Integrate market data feeds for Polymarket + Kalshi
2. Build condition_id graph (which markets share underlying conditions)
3. Run backtest on historical BTC threshold markets to validate signal frequency
4. Add to alert queue first; auto-execute only after validation

---

## References

- [arXiv:2508.03474](https://arxiv.org/abs/2508.03474) — Saguillo et al. main paper
- [Flashbots discussion](https://collective.flashbots.net/t/arbitrage-in-prediction-markets-strategies-impact-and-open-questions/5198)
- [Bawa Medium: 62% LLM false positive rate](https://medium.com/@navnoorbawa/combinatorial-arbitrage-in-prediction-markets-why-62-of-llm-detected-dependencies-fail-to-26f614804e8d)
