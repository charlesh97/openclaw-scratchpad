# octavi42/prediction-market-maker

**Source:** https://github.com/octavi42/prediction-market-maker

## What It Does
A **market-making strategy** that placed #2 in Paradigm's Prediction Market Challenge (110 iterations, 8 hours). This is NOT an arbitrage bot — it's a market maker that quotes bid/ask spreads on prediction markets, profiting from the spread between informed and uninformed flow. The key insight: market makers in prediction markets earn by capturing the spread, not by directional bets.

This is a useful complement to arb strategies: after you've identified a market is inefficient, market making lets you earn a consistent spread while waiting for the market to correct.

## Key Concepts from the Repo
- **LMSR (Logarithmic Market Scoring Rule)** market maker mechanics
- **Adverse selection**: informed traders know something — market makers must charge a spread to survive
- **Inventory risk**: holding positions overnight in fast-resolving markets is dangerous
- **Order sizing**: smaller orders = less adverse selection but lower earnings
- Ranked #2 out of all Paradigm Prediction Market Challenge participants

## Implementability: 3/5
- Paradigm challenge context shows it works at scale
- Conceptually important for understanding how prediction market microstructure actually works
- Not a direct arb strategy but crucial background for building the complete picture
- arXiv paper reference for deeper theory

## Risks
- Market making in prediction markets is complex — requires real-time position management
- Unlike crypto AMMs, prediction market maker risk is directional and event-driven
- The profitable spread may already be compressed by professional players

## Next Steps
1. Study the LMSR mechanics — foundational for understanding how Polymarket prices are set
2. Combine with arb bots: arb detects mispricing, market making earns spread while you hold the arb position
