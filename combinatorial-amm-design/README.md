# Designing Automated Market Makers for Combinatorial Securities

**Source:** https://arxiv.org/pdf/2411.08972

## What it does
Introduces the combinatorial swap operation problem for automated market makers in decentralized finance. Shows that it can be efficiently reduced to range update problems. In prediction markets, traders buy/sell securities based on future event outcomes — this paper designs AMMs that can handle combinatorial securities efficiently.

## Key Contributions
- Formal model for combinatorial securities in AMMs
- Efficient reduction of combinatorial swap to range update problems
- Design principles for market makers that avoid computational bottlenecks
- Arbitrage-free pricing conditions for combinatorial markets

## Why it matters
If prediction markets move toward combinatorial securities (baskets, bundles, conditional outcomes), the AMM mechanics change. Understanding this now positions us for the next generation of prediction market design.

## Implementability: 2/5
Theoretical paper with Solidity-style pseudocode. Not directly deployable but important for long-term strategy.

## Next Steps
1. Monitor if Polymarket or competitors adopt combinatorial securities
2. If so, implement the efficient swap mechanisms
