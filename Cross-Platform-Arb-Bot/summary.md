# Summary: Toward Black Scholes for Prediction Markets

**arXiv:2510.15205** | Oct 2025 (updated Apr 2026)

## One-Page TL;DR

Proposes a **logit jump-diffusion model** as the Black-Scholes equivalent for prediction markets — creating a standardized framework for quoting, hedging, and transferring belief risk.

### Key Contributions
1. **Logit jump-diffusion kernel** — treats probability p_t as a Q-martingale
2. **Calibration pipeline** — filters microstructure noise, separates diffusion from jumps via EM
3. **Derivative layer** — variance, correlation, corridor, and first-passage instruments
4. **Belief-volatility surface** — implied volatility analogue for prediction markets

### Why This Matters
- Prediction markets lack standardized pricing tools compared to options
- As institutional participation grows, standardized hedging instruments become essential
- This framework could enable market makers to quote options on prediction market outcomes

### Practical Takeaway
While this is theoretical, the calibration pipeline and noise-filtering techniques can improve our arbitrage detection by separating genuine mispricings from microstructure noise.
