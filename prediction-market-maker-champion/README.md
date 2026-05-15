# Prediction Market Maker — Paradigm Challenge #2

**Source:** https://github.com/octavi42/prediction-market-maker  
**Recommendation:** YES

## What It Does

A market-making strategy that placed #2 in Paradigm's Automated Research Hackathon (April 2026). The bot operates as a liquidity provider on binary YES/NO prediction markets, placing passive limit orders to capture the spread. Over 110 strategy iterations in 8 hours, it achieved a mean edge of $41.09 per simulation.

## Key Discoveries

### The Monopoly Regime (~60% of edge)
The single most valuable insight: when the market maker is the only liquidity provider at a given price level, they can quote wider spreads without losing fills. This alone was worth more than 100 parameter tweaks.

### Volatility-Adjusted Quote Filtering
The strategy dynamically sits out during high-volatility periods when adverse selection risk exceeds the expected spread capture. This prevents filling orders that are likely to be mispriced.

### Inventory Management (removing it = -$7 swing)
Active skew adjustment prevents catastrophic losses. When the bot accumulates too much of one side, it widens the opposite side's quote to rebalance.

### Sizing > Parameter Tweaking
Matching expected retail order flow outperformed complex pricing models. The strategy sizes orders proportional to expected fill probability rather than fixed amounts.

## Why It Matters

This is a complete case study in prediction market microstructure — the mechanics of quoting, adverse selection, inventory risk, and order sizing. The codebase is well-documented with a Jupyter notebook analysis of what worked and what failed across all 110 iterations. Directly applicable to building Polymarket market-making strategies.

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Simulated environment ≠ live | Medium | Core insights transfer to live |
| Polymarket FIFO order book | Low | Directly matches challenge setup |
| Competition from other MMs | Medium | Monopoly regime insight is structural |
| Adverse selection in news events | High | Volatility filtering handles this |

## Implementability: 5/5

Pure Python, Jupyter notebooks, clear mathematical framework. The strategy logic is about 200 lines of Python. No external dependencies beyond numpy/pandas. The edge sources are well-understood and transfer directly to Polymarket's actual order book.

## Next Steps

1. Port the strategy to live Polymarket using the CLOB API
2. Add real-time order book monitoring
3. Test monopoly regime detection in live markets
4. Implement the volatility-adjusted quoting with real data
