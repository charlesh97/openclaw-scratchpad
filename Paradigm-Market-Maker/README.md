# Paradigm Market Maker (octavi42)

**Source:** [octavi42/prediction-market-maker](https://github.com/octavi42/prediction-market-maker)
**Recommendation:** MEDIUM

## What It Does

A market-making strategy that placed **#2 in Paradigm's Prediction Market Challenge** (April 2026). 110 strategy iterations in 8 hours.

### Key Insights
- **Monopoly regime** (~60% of edge) — being the only maker on certain price levels
- **Volatility-adjusted quote filtering** — when to quote and when to sit out
- **Inventory management** — skew prevents catastrophic losses (removing it = -$7 swing)
- **Sizing matters** — match expected retail order flow

### Results
| Metric | Value |
|--------|-------|
| Placement | #2 of all submissions |
| Mean edge | $41.09 per simulation |
| Strategy versions | 110 |
| Dev time | 8 hours |

## Implementability: 4/5

- Python-based, clean implementation
- Comprehensive case study of prediction market microstructure
- Educational value is exceptional

## Risks
- Challenge was simulated — real Polymarket may differ
- Market making requires significant capital for quote flooding
- Adverse selection risk in live markets

## Next Steps
Study the monopoly regime finding — this is the single most valuable insight from the challenge.
