# Prediction Market Maker — Paradigm Challenge (#2 Place)

**Source:** https://github.com/octavi42/prediction-market-maker

## What it does
A complete case study in market making for prediction markets. Placed #2 in Paradigm's Prediction Market Challenge. Covers the mechanics of quoting, adverse selection, inventory risk, and order sizing across 110 iterative improvements over 8 hours of competition.

## Key Concepts Demonstrated
- **How market makers profit** — Capturing the spread between uninformed flow and true probability
- **Adverse selection** — Avoiding being picked off by informed traders
- **Inventory risk** — Managing directional exposure
- **Order sizing** — Optimal quote sizes given market depth
- **Competition dynamics** — Iterating strategies in real-time against other participants

## Why it matters
Pure market making is a different strategy from arbitrage, but it complements it. A market maker earns steady spread income while the arb bot captures dislocations. Combining both in one system diversifies revenue.

## Implementability: 4/5
Python + standard data science stack. Well-documented with iteration logs. The challenge format means the strategy is tested under competitive pressure, which validates the approach.

## Next Steps
1. Study the 110 iteration progression to understand what worked
2. Implement the core market making logic
3. Integrate with arb detection as a separate strategy module
