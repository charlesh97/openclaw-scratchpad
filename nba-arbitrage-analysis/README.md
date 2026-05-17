# Arbitrage Analysis in Polymarket NBA Markets

**Source:** arXiv:2605.00864 — https://arxiv.org/abs/2605.00864
**Authors:** Guang Cheng et al.
**Published:** April 2026

## Key Findings
Systematic empirical analysis of algorithmic arbitrage in Polymarket's NBA game markets:
- **75M+ limit order book snapshots** across 173 games
- **Single-market arb:** Only 7 executable in-game episodes (median duration: 3.6 seconds)
- **Combinatorial arb:** 290 active episodes, concentrated in final minutes
- **Median return:** 101 basis points per combinatorial arb
- **Liquidity bottleneck:** 76.9% of opportunities limited to 14.8 shares average size
- The theoretical "Middle" jackpot (both Moneyline + Spread pays) is never empirically realized

## Why It Matters
Confirms that while mispricings exist in Polymarket sports markets, they are **structurally bounded by liquidity** — risk-free extraction is strictly retail-scale. This validates the "limits to arbitrage" framework in decentralized markets.

## Implementability: 4/5
The methodology (continuous market state reconstruction from snapshots) is directly applicable to live monitoring. The finding that 76.9% of arb is <15 shares suggests focusing on high-frequency, small-size execution for sports arb.

## Next Steps
1. Replicate the methodology for live NBA market monitoring
2. Test automated small-size execution (under 15 shares) during final minutes
3. Extend analysis to UFC/NFL markets
