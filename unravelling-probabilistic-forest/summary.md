# Summary: Unravelling the Probabilistic Forest

## Research Question
What conditions give rise to arbitrage on Polymarket? Does arbitrage actually occur? Has anyone exploited it?

## Dataset
- Polymarket on-chain order book data: April 2024 — April 2025
- 86 million transactions, 17,218 conditions, 6,487,193 bettors

## Methodology
Used integer programming to detect arbitrage in condition sets. Heuristic reduction:
1. **Timeliness filter:** Only consider markets resolved within similar timeframes
2. **Topical similarity:** Group markets discussing related events
3. **Combinatorial relationships:** Connected conditions through shared tokens

## Key Results
- 7,051 conditions with single-market arbitrage (41%)
- $40M USD estimated realized profit from arb extraction
- Combinatorial arb more common but smaller per-opportunity
- Heuristic reduction is crucial: naive O(2^n) comparison is intractable for 17,218 conditions

## Implications
- Polymarket is far from efficient — $40M left on the table
- The arb detection algorithm can be automated and deployed
- Post-2026 dynamic fees may have reduced arb profitability
