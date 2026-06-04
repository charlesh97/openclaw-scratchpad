# Paper Summary: Unravelling the Probabilistic Forest

**arXiv:** 2508.03474  
**Authors:** Satheeshkumar et al.  
**Date:** August 5, 2025  

## Research Question
What types of arbitrage exist in prediction markets, how large are they, and are they being exploited?

## Methodology
- Full on-chain historical order book data from Polymarket
- 86M transactions, 17,218 conditions
- Integer programming approach for combinatorial arb detection
- Heuristic reduction via timeliness, topical similarity, and combinatorial structure
- DAG-based dependency modeling

## Key Results
| Metric | Value |
|--------|-------|
| Conditions with arbitrage | 7,051 / 17,218 (41%) |
| Total arb profits (12mo) | ~$40M |
| Combinatorial arb share | 0.24% of profits |
| LLM-dep. failure rate | 62% |
| Median market deviation | $0.60 (40% from efficient) |

## Two Types of Arbitrage

### Market Rebalancing (Simple Bundle Arb)
- YES + NO < $1.00
- $40M captured, now dominated by HFT bots
- Fun fact: the strategy that "guarantees profit" is what $40M was based on

### Combinatorial Arb (Cross-Market)
- Exploits pricing inconsistencies between logically dependent conditions
- Requires integer programming optimization
- 62% of LLM-identified dependencies don't actually generate profit
- But the remaining 38% are underexploited

## Practical Takeaways
1. Simple bundle arb is dead for retail (too much competition)
2. Combinatorial arb is the next frontier but mathematically hard
3. Integer programming approaches can detect profitable cross-market opportunities
4. Failed dependencies (62%) mean high research cost — need systematic screening
5. Market is NOT efficient — $0.60 median deviation is massive

## Relevance to Our Research: Very High
This is the foundational arbitrage paper. Combinatorial arb is the most promising uncrowded strategy direction.
