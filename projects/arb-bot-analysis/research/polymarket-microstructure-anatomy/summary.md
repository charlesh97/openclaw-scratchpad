# Paper Summary: The Anatomy of a Decentralized Prediction Market

**arXiv:** 2604.24366  
**Authors:** (Anonymous, peer-reviewed)  
**Date:** April 27, 2026  

## Research Question
What are the empirical microstructure properties of Polymarket, the largest on-chain prediction market?

## Methodology
- 30 billion continuous tick-level order-book events over 52 days
- Joined to authoritative on-chain trade records
- Pre-registered stratified panel of 600 markets
- 8 stylized fact tests

## Key Results
| Fact | Finding |
|------|---------|
| Spread premium | Longshot outcomes have wider spreads |
| Depth profile | Uniform, not top-of-book concentrated |
| Block clock | No alignment effect |
| Maker diversity | Broad base, concentrated top |
| Category spreads | Significant cross-category variation |
| Ingestion delay | Median <50ms, tail >1s |
| Self-trading | 1% median, 22% upper tail |
| Wash trade | Below CEX benchmarks |

## Practical Takeaways
1. Sub-50ms median delay means fast bots have an edge but it's not untouchable
2. Uniform depth means you can execute larger orders than a top-of-book analysis suggests
3. Category matters — some markets are more liquid and tighter than others
4. Wash trading exists at low levels but some markets may have inflated volume
5. Short-term crypto markets are where most microstructure action happens

## Relevance to Our Research: High
Provides the empirical foundation needed to build realistic strategy simulations.
