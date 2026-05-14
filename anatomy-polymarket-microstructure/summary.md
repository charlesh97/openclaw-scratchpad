# Summary: The Anatomy of a Decentralized Prediction Market

**arxiv.org/abs/2604.24366**

## One-Page TL;DR

The most comprehensive microstructure analysis of Polymarket ever conducted, using 30B order-book events across 600 markets.

**8 Stylized Facts:**
1. Longshot bias in spreads (worse odds = worse liquidity)
2. Geometric depth grid (not just top-of-book)
3. No block-clock effects
4. Many makers but a few dominate
5. Sports spreads > crypto spreads
6. ~50ms median data delay
7. 1-22% wash trading
8. Effective spreads vary by category

**For bot builders:** This paper gives you the empirical ground truth. Crypto markets offer the best liquidity. Expect 50ms+ data lag. Wash trading can inflate volume signals by up to 22%. These constraints should drive market selection, execution timing, and signal filtering decisions.
