# Toward Black Scholes for Prediction Markets

**Source:** https://arxiv.org/abs/2510.15205 (Oct 2025, updated Apr 2026)

**Authors:** Shaw Dalen et al.

## Summary

Proposes a unified stochastic kernel for prediction markets — analogous to what Black-Scholes did for options. Introduces a **logit jump-diffusion with risk-neutral drift** that treats the traded probability p_t as a Q-martingale.

### Key Contributions
- **Belief volatility surface** — analogous to implied volatility in options
- **Jump intensity estimation** — separates diffusion from jumps using expectation-maximization
- **Microstructure noise filtering** — calibration pipeline for real data
- **Derivative layer** — variance, correlation, corridor, and first-passage instruments

### Practical Application
Market makers can finally quote, hedge, and transfer belief risk using standardized tools — including cross-venue hedging across Polymarket.

## Implementability: 2/5

Highly theoretical. The practical applications (belief volatility surface, hedging instruments) are valuable but require significant implementation work. Not directly usable as-is.

## Recommendation: MEDIUM

Important theoretical foundation. The logit jump-diffusion model could inform our risk management and position sizing, but this is longer-term research.
