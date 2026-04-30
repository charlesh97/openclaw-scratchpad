# Intra-Market Rebalancing Arbitrage (IMRA)

## What It Does

Detects and exploits price-sum violations within single-condition Polymarket markets.

In a binary prediction market, YES and NO shares for the same condition should sum to exactly $1.00 (since one must resolve true). When the market order book shows YES_bid × YES_ask and NO_bid × NO_ask at prices that violate this invariant, a risk-free arbitrage exists.

**Example:** A market asks "Will BTC close above $100k by June 1?"
- Best YES ask: $0.70 (implying 70% probability)
- Best NO ask: $0.26 (implying 26% probability)
- Sum = $0.96 < $1.00 → buy both for $0.96, guaranteed to receive $1.00 at resolution → **4% risk-free return**

## How It Differs From Existing Algorithms

| Algorithm | What It Catches | This Algorithm |
|---|---|---|
| CMRA Detector | Multi-condition mispricings across markets | **Single-condition intra-market violations** |
| Semantic Identity Arb | Same event priced differently across platforms | Same platform, same condition, orderbook violation |
| KL Latency Arb | Price moves from CEX → prediction market | No CEX dependency — pure orderbook arbitrage |

This is the most "classical" form of prediction market arb — the closest analog to arbitrage in traditional exchanges. It operates entirely within Polymarket's CLOB with no external data dependency.

## Architecture

```
Polymarket CLOB API
        ↓
 Single-Condition Market Filter  ← filters out multi-condition markets
        ↓
 Best Bid / Best Ask for YES and NO
        ↓
 YES_ask + NO_ask < 1.0 ?       → BUY both legs (guaranteed spread)
 YES_bid + NO_bid > 1.0 ?       → SELL both legs (reverse arb)
        ↓
 Execution Engine (post orders, await fills)
        ↓
 Profit锁定 at resolution
```

## Implementation Difficulty

**3/5** — Moderate. Requires:
- Polymarket CLOB API access (graphql endpoint)
- Order book parsing and best bid/ask extraction
- Single-condition market identification (vs. multi-condition)
- Order posting and fill tracking

## Key Findings From Research

- IMDEA paper (arXiv:2508.03474) documents this as **Type 1 Rebalancing Arbitrage**
- $40M confirmed extracted across all arb types (April 2024–April 2025)
- Average opportunity window: **2.7 seconds** (down from 12.3s in 2024)
- **73% of profits captured by sub-100ms execution**
- 78% of low-volume opportunities fail due to execution inefficiency

## Opportunity Characteristics

- **Spread capture:** Typically 0.5%–5% per opportunity
- **Frequency:** More common than cross-platform arb; every YES/NO pair is a potential trade
- **Capital efficiency:** Two-leg position requires capital on both sides
- **Risk:** Resolution risk is zero; execution risk is non-trivial (fills not guaranteed)
- **Liquidity constraint:** Only viable in markets with sufficient depth on both legs

## Risks

1. **Execution risk:** Polymarket is CLOB — orders sit in book. No guarantee of fill at detected price
2. **Latency risk:** 2.7s median window; human-traded, uncompetitive vs. bot-filled books
3. **Fee risk:** Trading fees may exceed narrow spread captures
4. **Market closure risk:** Market resolved or paused mid-position
5. **Adverse selection:** High-frequency market makers may front-run visible arb opportunities

## Next Steps

1. Connect to Polymarket GraphQL API — fetch single-condition markets and live orderbook
2. Build orderbook monitor: track best bid/ask for YES and NO simultaneously
3. Calculate spread: `YES_ask + NO_ask - 1.0` in real time
4. Implement position sizing: conservative sizing since fills are not guaranteed
5. Paper trade first — validate fill rates before live capital
6. Add execution tier: if fill not received in X ms, cancel and skip

## References

- [arXiv:2508.03474](https://arxiv.org/abs/2508.03474) — Saguillo et al., "Unravelling the Probabilistic Forest" (August 2025)
- [Polymarket API](https://docs.polymarket.com/) — CLOB orderbook and market data
- [IMDEA Flashbots Post](https://collective.flashbots.net/t/arbitrage-in-prediction-markets-strategies-impact-and-open-questions/5198) — TL;DR of paper findings
