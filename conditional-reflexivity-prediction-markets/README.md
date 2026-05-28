# Conditional Reflexivity & Prediction Markets

**Source:** [arXiv 2604.24147](https://arxiv.org/abs/2604.24147) (April 2026)

## What It Does

Argues that prediction markets under certain conditions function as **coordination mechanisms** rather than pure forecasting devices. Introduces the **Signal Credibility Index (SCI)** — a microstructure-grounded criterion for predicting when price moves acquire behavioral traction.

## Key Concepts

- **Conditional Reflexivity:** Market probabilities don't just forecast the future — they can actively shape it (voters, donors, journalists, institutions respond to market signals)
- **Signal Credibility Index (SCI):** Combines variance ratio VR(6), two-sidedness diagnostic, and trader-concentration adjustment
- **Key empirical finding:** The most visible prediction market (Polymarket 2024 election) produced the **least accurate forecasts** — cross-platform comparison shows systematic decoupling of social authority from epistemic robustness
- **Regulatory implications:** Deregulatory trajectory (2025-2026) may improve liquidity while degrading epistemic quality

## Why It Matters

For trading bot development, the reflexivity insight has practical implications:
1. **Self-fulfilling price moves** in high-profile markets create momentum that bots can exploit
2. **Signal quality varies** — the SCI framework can filter which markets have genuine information vs. coordination effects
3. **Cross-platform decoupling** creates arbitrage opportunities when Polymarket prices diverge from Kalshi/Robinhood due to different coordination dynamics

## Risks

- Purely theoretical framework — no deployable trading code
- The reflexivity effect is strongest in political markets, not crypto/sports
- SCI requires microstructure data not easily available in real-time
- Academic paper, not a trading system

## Implementability: 2/5

Important conceptual framework for understanding market dynamics, but not directly deployable. Valuable for strategic positioning rather than execution.

## Next Steps

1. Implement the SCI as a market-quality filter
2. Test whether SCI-identified "coordination" markets exhibit momentum patterns
3. Compare cross-platform decoupling during major political events
4. Use reflexivity framework to improve entry/exit timing
