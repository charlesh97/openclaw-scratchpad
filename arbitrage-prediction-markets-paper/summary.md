# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Authors:** Oriol Saguillo, V. Ghafouri, L. Kiffer, G. Suarez-Tangil (IMDEA Networks Institute)
**Source:** https://arxiv.org/abs/2508.03474 (also: AFT 2025 proceedings)
**Published:** August 2025
**Recommendation:** YES (key arbitrage research)

## Summary

First large-scale analysis of arbitrage on Polymarket. Addresses three questions: (Q1) What conditions give rise to arbitrage? (Q2) Does arbitrage actually occur? (Q3) Has anyone exploited these opportunities?

### Two Forms of Arbitrage Discovered:

1. **Market Rebalancing Arbitrage** — Within a single market/condition. When YES + NO prices don't sum to $1.00, buying both guarantees profit.
2. **Combinatorial Arbitrage** — Across multiple markets. When logically related conditions are priced inconsistently.

### Key Finding

**$40M USD of realized profit extracted** from arbitrage on Polymarket. The paper uses a heuristic-driven reduction strategy to overcome the O(2^(n+m)) complexity of naive pairwise comparison across all conditions.

### Methodology

Applied to 17,218 total conditions. Found 7,051 conditions had single-market arbitrage. Used timeliness, topical similarity, and combinatorial relationships with expert validation to reduce the search space.

### Relevance

Directly applicable to building arbitrage detection strategies. The $40M extracted profit demonstrates arb is real and profitable. The combinatorial arb detection methodology can be implemented in our bots.

### Key Takeaway

Arbitrage on Polymarket is real and profitable ($40M extracted). Two distinct types exist. Combinatorial arb is harder to detect but offers larger opportunities.
