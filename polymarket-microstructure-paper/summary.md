# The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book

**Authors:** Philipp Dubach et al.
**Source:** https://arxiv.org/abs/2604.24366
**Published:** April 27, 2026
**Recommendation:** YES (foundational reference)

## Summary

This paper studies Polymarket's microstructure using a continuous tick-level archive of the public order-book feed — **30 billion events over 52 days** — joined to the authoritative on-chain trade record. Analyzes 600 markets on a pre-registered stratified panel.

### 8 Stylized Facts:

1. **Longshot spread premium** — Wider spreads for extreme probabilities
2. **Depth profile closer to uniform** than top-of-book concentrated
3. **Null block-clock alignment effect** — No timing patterns around blocks
4. **Broad maker-wallet diversity** with a concentrated tail (few wallets dominate)
5. **Category-conditional effective-spread differences**
6. **Sub-50ms median ingestion delay** with multi-second tail
7. **Self-counterparty wash share**: median 1%, upper tail 22% (well below crypto norm of 25-70%)
8. **Cross-sectional depth profile** explained by duration, price, volume — no residual time-to-close effect

### Critical Measurement Finding:

Trade direction inferred from Polymarket's public order-book feed agrees with on-chain ground truth on only **~59% of buckets** — well below the ~80% Lee-Ready accuracy on Nasdaq. This means microstructure research on Polymarket **must source trade direction from on-chain OrderFilled events**.

### Relevance

Essential reading for anyone building prediction market trading bots. The 8 stylized facts inform strategy design, and the trade-direction finding is critical for avoiding measurement error in backtesting.

### Key Takeaway

Polymarket's microstructure is fundamentally different from traditional markets — the order book is shallower, trade direction is harder to infer, and wash trading is lower but non-trivial.
