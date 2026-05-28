# Arbitrage Analysis in Polymarket NBA Markets

**Source:** [arXiv 2605.00864](https://arxiv.org/abs/2605.00864) — Guang Cheng, Jiaxin Yang et al. (April 2026)

## What It Does

Conducts a systematic empirical analysis of algorithmic arbitrage within Polymarket's NBA game markets using **75 million limit order book snapshots across 173 games**. This is the first large-scale, tick-level empirical study of arbitrage feasibility on Polymarket.

## Key Findings

- **Single-market arbitrage is extremely rare:** Only 7 executable in-game episodes found — median duration just 3.6 seconds
- **Combinatorial arbitrage is more common:** 290 active episodes, but overwhelmingly concentrated in the final minutes of live play
- **Combinatorial execution yields ~101 bps median return** — statistically meaningful but structurally limited
- **The theoretical "Middle" jackpot is never empirically realized** — the $1.00 portfolio is a myth in practice
- **76.9% of combinatorial opportunities are constrained** to an average executable size of just 14.8 shares
- **Shallow order book depth** is the primary bottleneck — liquidity shallowness confines risk-free extraction to retail scale

## Why It Matters

This paper empirically grounds the limits-to-arbitrage framework (Shleifer & Vishny 1997) in decentralized prediction markets. It confirms what practitioners already suspect: arbitrage exists but is structurally bounded by liquidity. For bot development, this means:
1. Simple YES/NO bundle arbitrage is effectively competed away
2. Combinatorial cross-market arbitrage has more signal but smaller executable sizes
3. Any profitable arb strategy must account for shallow book depth as the binding constraint

## Risks

- Findings are NBA-specific (high volatility, event-driven microstructure)
- Order book depth on Polymarket may improve over time, changing the constraints
- Results assume the cost of gas + fees doesn't exceed arb returns
- Paper identifies the problem more than providing a deployable solution

## Implementability: 3/5

Excellent empirical reference for strategy parameterization, but no standalone code to deploy. The findings inform position sizing and opportunity selection rather than providing an executable strategy.

## Next Steps

1. Validate findings against BTC/ETH 15-min markets (more volume, different microstructure)
2. Implement the combinatorial detection algorithm from the paper
3. Build a real-time scanner based on the paper's methodology
4. Compare executable sizes across market types (sports vs. crypto)
