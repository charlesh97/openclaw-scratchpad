# Summary: Unravelling the Probabilistic Forest

**Authors:** Oriol Saguillo et al.
**Published:** August 2025
**arXiv:** 2508.03474

## TL;DR
Two types of arbitrage exist on Polymarket: (1) Market Rebalancing — buying YES+NO when they sum < $1, and (2) Combinatorial Arb — exploiting price inconsistencies across logically linked markets. Both are detectable and exploitable systematically.

## Data
- On-chain order book data from Polymarket
- Multiple event categories (sports, politics, crypto)

## Key Numbers
- Market Rebalancing arb occurs regularly across thousands of markets
- Combinatorial arb requires solving a constraint graph across related conditions
- Arb windows typically persist for minutes, not hours

## Relevance to Vega Research
High. This is the closest academic paper to our arb bot project. The combinatorial arb framework is directly implementable and complements the cross-exchange arb from the Kalshi bot.
