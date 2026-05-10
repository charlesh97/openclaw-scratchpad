# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474
**Type:** Research paper (August 2025)
**Recommendation:** YES — Top of Email

## What It Does

First large-scale empirical analysis of arbitrage on Polymarket. The paper defines two distinct arbitrage mechanisms:

### 1. Market Rebalancing Arbitrage (Intra-market)
- Occurs within a single market/condition
- Exploits price movements that temporarily push YES + NO away from $1.00 equilibrium
- Cleaner to execute — single venue, no cross-platform risk

### 2. Combinatorial Arbitrage (Inter-market)
- Spans across multiple markets
- Exploits dependencies between conditions where related markets misprice relative to each other
- More complex: requires modeling conditional relationships across the full "probabilistic forest"

The authors used **on-chain historical order book data** to identify when opportunities existed AND when they were actually executed — giving a ground-truth picture of which gaps were filled vs. missed.

## Why It Matters

This is the conceptual foundation for any arb system you build. Before writing a single line of code, you need to internalize:
- The two arbitrage modes and which one your bot should target
- The actual magnitude and frequency of opportunities (not the Twitter-thread fiction)
- Which gaps get filled and which persist (liquidity constraints, not just speed)

## Key Structural Insights

- Combinatorial arb opportunities are harder to detect and execute — requires modeling market relationships
- Even clearly-identified mispricings don't fully correct due to execution frictions (Shleifer-Vishny limits-to-arbitrage)
- Order book depth is the binding constraint on profitable execution

## Implementability: 5/5

This is a **must-read framework paper**, not a code library. Read the paper, internalize the two-mode framework, then build your detection logic accordingly.

**Next steps:**
1. Read arxiv.org/abs/2508.03474 (paper PDF)
2. Model your arb detection as two separate pipelines: (a) single-market rebalancing and (b) cross-market combinatorial
3. Use the order-book depth analysis to calibrate your maximum position sizing