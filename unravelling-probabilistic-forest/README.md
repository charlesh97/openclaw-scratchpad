# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** arXiv:2508.03474 — https://arxiv.org/abs/2508.03474
**Authors:** Oriol Saguillo, Vazha Ghafouri, Lucianna Kiffer, Guillermo Suarez-Tangil (IMDEA Networks Institute)
**Published:** August 2025 | **Conference:** AFT 2025

## Key Findings
- **$40M+ profit extracted** from Polymarket arbitrage between April 2024 and April 2025
- Analyzed **86 million transactions** across **17,218 conditions**
- Two types of arbitrage discovered:
  1. **Market Rebalancing Arbitrage** (intra-market): within a single condition
  2. **Combinatorial Arbitrage** (inter-market): across multiple related markets
- **7,051 conditions** had single-market arbitrage — 41% of all conditions
- Used heuristic-driven reduction (timeliness, topical similarity, combinatorial relationships) to scale beyond O(2^n) analysis

## Why It Matters
This is the first large-scale empirical analysis of arbitrage on Polymarket. The $40M figure quantifies the opportunity. The combinatorial arb detection algorithm is the key deliverable — can be replicated and automated.

## Implementability: 4/5
The combinatorial arb detection heuristic is implementable. The paper's algorithm for detecting mispriced condition sets can be turned into a real-time scanner. Main limitation: $40M was over 1 year — current opportunity may be lower as competition has increased.

## Next Steps
1. Implement the combinatorial arb detection algorithm
2. Run on live Polymarket data
3. Benchmark arb profitability before and after Polymarket's dynamic fee changes (2026)
