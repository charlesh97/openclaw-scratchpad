# The Anatomy of a Decentralized Prediction Market

**Source:** https://arxiv.org/abs/2604.24366 (April 2026)

## Summary

The most detailed microstructure study of Polymarket ever conducted. Analyzed **30 billion events** over 52 days, joined to the authoritative on-chain trade record.

### Key Findings (8 Stylized Facts)

1. **Longshot spread premium** — wider spreads on low-probability outcomes
2. **Depth profile closer to uniform** than top-of-book — liquidity is distributed
3. **Null block-clock alignment** — no time-based pattern in liquidity
4. **Broad maker-wallet diversity** with a concentrated tail
5. **Category-conditional effective-spread differences** — sports markets tighter than crypto
6. **Sub-50 ms median archive-ingestion delay** with a multi-second tail
7. **Self-counterparty wash share** — median 1%, upper tail 22%
8. **Liquidity constraints limit arbitrage execution**

### Relevance to Our Bot
Provides the empirical foundation for understanding Polymarket's microstructure. The wash trading finding (1% median, 22% tail) is important for trade signal filtering. The category-dependent spreads help choose target markets.

## Implementability: 2/5
Research paper, not software. But the stylized facts directly inform trading strategy design — especially market selection, liquidity assessment, and wash trade filtering.
