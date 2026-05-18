# Toward Black Scholes for Prediction Markets: A Unified Kernel and Market Maker's Handbook

**Source:** https://arxiv.org/abs/2510.15205
**Published:** October 2025 (v2: April 2026)

## Summary

Proposes a **logit jump-diffusion** framework as the "Black-Scholes for prediction markets" — a unifying stochastic kernel for pricing, quoting, and hedging prediction market risk.

### Key Contributions

1. **Logit Jump-Diffusion Kernel**: Treats traded probability p_t as a Q-martingale, exposing belief volatility, jump intensity, and dependence as quotable risk factors
2. **Calibration Pipeline**: Filters microstructure noise, separates diffusion from jumps via expectation-maximization, enforces risk-neutral drift
3. **Derivative Layer**: Defines variance, correlation, corridor, and first-passage instruments — analogous to volatility products in options markets

### Results
- Reduces short-horizon belief-variance forecast error vs. diffusion-only baselines
- Provides an "implied volatility" analogue for prediction markets
- Works on both synthetic risk-neutral paths and real event data

## Why It Matters
If prediction markets are to scale to institutional levels, they need standardized pricing tools — just as options needed Black-Scholes. This paper provides that framework.

## Implementability: 1/5
Highly mathematical (stochastic calculus, EM algorithm). Best used as conceptual foundation for pricing models, not directly deployable.

## Next Steps
1. Study the jump-diffusion calibration methodology
2. Implement simplified belief-volatility surface for our portfolio
3. Use the framework for sizing positions based on belief volatility
