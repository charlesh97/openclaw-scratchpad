# The Anatomy of a Decentralized Prediction Market

**Source:** https://arxiv.org/abs/2604.24366

## What it does

The first tick-level microstructure analysis of Polymarket using 30 billion WebSocket order book events over 52 days, joined to the on-chain trade record. Pre-registered stratified panel of 600 markets.

## Eight stylized facts discovered

1. **Longshot spread premium:** Longshot outcomes have wider spreads
2. **Uniform depth grid:** Depth distribution is more uniform than traditional markets
3. **Null block-clock alignment:** No time-based trading patterns at block boundaries
4. **Broad maker diversity with concentrated tail:** Many small MMs, few dominant ones
5. **Category-conditional spreads:** Sports markets have different spreads than crypto
6. **Sub-50ms ingestion lag:** With multi-second tail for some events
7. **Self-counterparty wash share:** Median 1%, upper tail 22%

## Why it matters

Essential empirical baseline for any Polymarket bot. Knowing the microstructure facts (e.g., longshot premium, median 1% wash trade) helps design strategies that account for Polymarket's unique market structure.

## Implementability: 2/5

Research paper, no code. But the empirical findings directly inform strategy design.

## Risks

- N/A (empirical study)

## Next Steps

1. Use longshot findings to calibrate position sizing
2. Account for wash trading in volume-based signals
3. Factor category-conditional spreads into market selection
