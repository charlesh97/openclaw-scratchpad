# Toward Black Scholes for Prediction Markets (arXiv:2510.15205)

**Source:** https://arxiv.org/abs/2510.15205
**Author:** Shaw Dalen
**Published:** October 2025

## Key Finding
Proposes a **logit jump-diffusion kernel** for prediction markets — analogous to Black-Scholes for options. The model treats the traded probability p_t as a Q-martingale and exposes belief volatility, jump intensity, and dependence as quotable risk factors.

## Why It Matters
This is the first unified stochastic framework for prediction market pricing. If adopted, it would enable:
- **Implied belief volatility surfaces** (like implied vol for options)
- **Standardized quoting** of belief risk
- **Hedging cross-market correlation risk**
- **Calibration pipeline** that filters microstructure noise and separates diffusion from jumps

## Key Innovations
1. Logit jump-diffusion (probability-space equivalent of Black-Scholes)
2. Risk-neutral drift enforcement via EM algorithm
3. Belief-volatility surface analogous to option IV surfaces
4. Derivative layer: variance, correlation, corridor, first-passage instruments

## Implementability: 3/5
**MEDIUM** — mathematically sophisticated. The calibration pipeline (expectation-maximization for jump/diffusion separation) requires significant implementation effort. However, the core insight (belief volatility surface) is directly useful for market making.

## Next Steps
1. Implement the logit jump-diffusion calibration for Polymarket price series
2. Build a belief volatility surface for BTC 15-min markets
3. Use the surface to adjust market-making spread dynamically
