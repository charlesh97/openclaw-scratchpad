# Toward Black-Scholes for Prediction Markets

**Source:** https://arxiv.org/abs/2510.15205  
**Recommendation:** MEDIUM

## Summary

Proposes a logit jump-diffusion stochastic kernel for prediction markets — analogous to what Black-Scholes provided for options markets. Treats the traded probability p_t as a Q-martingale and exposes belief volatility, jump intensity, and cross-event dependence as quotable risk factors.

## Key Contributions

1. **Logit Jump-Diffusion Model** — A unifying stochastic framework for prediction market probabilities
2. **Calibration Pipeline** — Filters microstructure noise, separates diffusion from jumps via EM algorithm
3. **Risk-Neutral Drift Enforcement** — Ensures prices are consistent with no-arbitrage
4. **Belief Volatility Surface** — Analogous to implied volatility surface in options
5. **Derivative Layer** — Variance, correlation, corridor, and first-passage instruments

## Empirical Results

Reduces short-horizon belief-variance forecast error relative to diffusion-only and probability-space baselines. Supports both causal calibration and economic interpretability.

## Why It Matters

If this framework gains adoption, it would revolutionize prediction market pricing — providing standardized tools for quoting, hedging, and transferring belief risk. Directly relevant to designing market-making strategies that account for jump risk and volatility clustering.

## Implementability: 1/5 (Theoretical framework)

Purely theoretical at this point. The framework could guide risk model design but no production-ready implementation exists.

## Next Steps

- Implement the EM-based diffusion/jump separation
- Build belief volatility surface for active Polymarket markets
- Test the derivative layer on live data
