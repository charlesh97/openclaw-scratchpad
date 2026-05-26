# Prediction Market Maker — Paradigm Challenge #2

**Source:** https://github.com/octavi42/prediction-market-maker  
**Author:** octavi42 (octavicristea)  
**Language:** Python  
**License:** MIT  
**Status:** Case study (Paradigm Hackathon, April 2026)

## What It Does

A market-making strategy that placed **#2 in Paradigm's Prediction Market Challenge** (April 9, 2026), scoring $41.09 mean edge per simulation after 110 strategy iterations over 8 hours. Designed for a simulated binary YES/NO limit order book.

### Key Discoveries

1. **The Monopoly Regime** (~60% of edge) — When your quote is the only one on one side of the book, you can capture outsized spreads. The single most impactful insight.
2. **Volatility-Adjusted Quote Filtering** — Knowing when NOT to quote is as important as knowing when to quote.
3. **Inventory Management** — Skew prevents catastrophic losses. Removing it causes a -$7 edge swing.
4. **Sizing Matters More Than You Think** — Must match expected retail order flow.

## Why It Matters

- **Pure education** — Complete case study in prediction market microstructure: quoting, adverse selection, inventory risk, order sizing
- **Proven edge sources** — Monopoly regime and volatility filtering are directly applicable to Polymarket MM
- **Quantitative framework** — Edge-based scoring, not P&L-based, measures pricing skill directly

## Risks

- Simulated environment — real Polymarket has different fee structure, latency, and competition
- Monopoly regime less common on liquid Polymarket markets
- Hackathon time pressure — needs hardening for production

## Implementability: 3/5

- Clean, well-documented Python code
- Educational value is very high
- Needs adaptation for real Polymarket CLOB (different from FIFO LOB)
- Strategies can be extracted as modules for our larger bot

## Next Steps

1. Study the monopoly regime detection logic for our MM strategies
2. Adapt volatility-adjusted quoting for Polymarket's CLOB
3. Integrate inventory skew management into our risk engine
