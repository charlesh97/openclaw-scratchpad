# PIRAP Framework — 1-Page Summary

**Paper:** Resolution-Aware Perpetual Futures on Binary Prediction Markets: An Empirical Risk-Design Framework Using Polymarket Data
**arXiv:** 2605.10400 (May 11, 2026)
**Authors:** Maksym Nechepurenko et al.
**Code:** https://github.com/ForesightFlow/event-linked-perps

## The Problem

Can we create perpetual futures contracts that track prediction market probabilities (e.g., "Trump wins 2028") through resolution? Standard perpetual frameworks fail because prediction markets have:
- Bounded outcomes (0 or 1, no continuous price discovery)
- Terminal jumps (price snaps to 0 or 1 at resolution)
- Thin order books near resolution

## The Solution: PIRAP

Six-component framework:

| Component | Purpose |
|-----------|---------|
| Index Estimator | Composite of mid, depth-weighted mid, time-decayed VWAP |
| Jump-Aware Tiered Margin | Higher margins near resolution to absorb terminal jumps |
| Leverage Compression | Automatically reduces leverage as resolution approaches |
| Resolution-Aware Funding | Boundary-correction to prevent price manipulation |
| Multi-Stage Halt Protocol | Prevents death spirals in the final hour |
| Eligibility Framework | Only high-liquidity markets qualify |

## Key Results

- 5 pre-registered "floors" tested on 13,298 Polymarket markets
- **PASS:** Boundary depth asymmetry, terminal-jump magnitude, halt liquidation reduction (-80%)
- **FAIL:** Bad-debt frequency (+2.4%), drawdown targets, median PnL floor
- **Conclusion:** Framework is **not deployment-ready** — but the halt-vs-margin scope distinction is a genuine contribution

## Why It Matters for Our Bot

The halt protocol and margin mechanics could be adapted for our arbitrage bot's risk management. The paper also documents Polymarket's oracle infrastructure (MOOV2 transition) which affects settlement timing — useful for our execution timing.
