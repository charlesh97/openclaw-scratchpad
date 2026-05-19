# Summary: Unravelling the Probabilistic Forest

**arXiv:2508.03474** | IMDEA Networks Institute | Aug 2025

## One-Page TL;DR

This paper analyzed **86M transactions** on Polymarket over one year and found that **41% of all markets** had exploitable pricing inconsistencies (YES+NO ≠ $1.00). Between April 2024-April 2025, traders extracted **~$40M in arbitrage profits**.

### Two Types of Arbitrage

1. **Market Rebalancing** (41% of markets): Buying YES and NO when they sum < $1.00
2. **Combinatorial** (cross-market): Exploiting price inconsistencies between related conditions

### Key Insight

The authors developed a heuristic reduction strategy that makes combinatorial arbitrage detection computationally tractable, reducing O(2^(n+m)) comparisons to linear-time filtering through timeliness, topical similarity, and combinatorial relationship filters.

### Why This Matters for Our Bot

This is the definitive empirical study of Polymarket arbitrage. It proves the opportunity is real and quantifiable. The heuristic framework gives us a blueprint for building our own detection engine.
