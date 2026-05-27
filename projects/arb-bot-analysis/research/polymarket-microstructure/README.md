# Polymarket Microstructure Analysis

**Source:** https://arxiv.org/abs/2604.24366 — "The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book" (Dubach et al., 2026)

## What It Does

This paper presents the first comprehensive microstructure analysis of Polymarket using a continuous tick-level archive of the public order-book feed — **30 billion events over 52 days** — joined to the authoritative on-chain trade record. The study covers a pre-registered stratified panel of **600 markets**.

### Eight Stylized Facts

1. **Longshot spread premium** — spreads widen for extreme probabilities
2. **Uniform-like depth profile** — depth is spread across price levels, not concentrated at top-of-book
3. **Null block-clock alignment** — no meaningful clustering around block production
4. **Broad maker diversity with concentrated tail** — many unique wallets, but a few dominate
5. **Category-conditional effective spreads** — spreads differ by market category (sports vs. politics vs. crypto)
6. **Sub-50ms median ingestion delay** — but a heavy multi-second tail
7. **Self-counterparty wash share** — median 1%, 22% upper tail (well below centralized crypto venues)
8. **Cross-sectional depth explained by duration, price, volume** — no residual time-to-close effect

### Critical Measurement Result

Trade direction inferred from Polymarket's public order-book feed agrees with on-chain ground truth on only **~59% of buckets** (panel mean 0.615), well below the ~80% Lee-Ready accuracy on Nasdaq. The effective half-spread **changes sign** between feed- and on-chain trade directions on 67%/50% of markets.

## Why It Matters

This is the definitive microstructure reference for building algorithms on Polymarket. Key takeaways:

- **Cannot rely on order-book trade direction** — must use on-chain `OrderFilled` events
- **Depth is spread, not concentrated** — large orders move prices significantly
- **Wash trading is low** — real volume, not fabricated activity
- **Replication package available** — authors released code to reproduce all results

## Risks

- Snapshot analysis, not a trading strategy — requires implementation
- Data from late 2025 — market structure may have evolved
- Focuses on descriptive statistics, not predictive signals

## Implementability: 3/5

The main value is methodological: the paper tells you **how** to build on Polymarket correctly (use on-chain data for trade direction, understand depth profiles). The replication package at https://github.com/philippdubach/polymarket-microstructure provides the data pipeline.

## Next Steps

1. Download the replication package and adapt the data pipeline for our bot
2. Implement on-chain trade direction inference (OrderFilled events)
3. Build depth-profile models for execution cost estimation
4. Use the "longshot spread premium" finding to improve quoting strategy
5. Monitor for changes since the study period
