# ForesightFlow: An Information Leakage Score Framework for Prediction Markets

**Source:** arXiv:2605.00493 (May 2026) — https://arxiv.org/abs/2605.00493
**Recommendation:** MEDIUM

## Summary

ForesightFlow introduces an Information Leakage Score (ILS) framework for detecting informed (insider) trading on decentralized prediction markets. The score measures what fraction of the terminal information move was priced in before the public news event.

## Key Concepts

- **ILS Score**: Quantifies information leakage pre-event. High ILS = more front-running.
- **Resolution-Anchored vs Event-Anchored**: The paper demonstrates that proxy quality for event timestamps is a binding constraint — using resolution timestamps can give misleading results.
- **Deadline-ILS Extension**: For markets that resolve at deadlines (not tied to specific news events), the authors extend the framework with exponential hazard baselines.

### Three Key Empirical Findings

1. Resolution-anchored proxies do NOT separate event-resolved markets from controls
2. Article-derived timestamps shift ILS scores dramatically vs proxies
3. All 24 documented Polymarket insider cases fall outside original ILS scope — motivating the deadline-ILS extension

## Why It Matters

For arbitrage and trading strategy development, understanding the information leakage landscape helps:

- **Identify where alpha resides**: Are you competing against insiders or retail? ILS scores reveal market categories with higher information asymmetry.
- **Market selection**: Choose markets with lower ILS for pure arbitrage strategies
- **Alerting**: Build ILS monitors to detect when markets might be moving on non-public information

## Risks

- Highly theoretical — not a directly deployable trading strategy
- Requires event resolution data that may not be available programmatically
- The 24-case auditor study found no scope-matching insider cases in the ILS framework

## Implementability: 2/5

Research framework, not a trading strategy. Useful as a monitoring/analytics reference but requires significant engineering to operationalize.

## Next Steps

1. Monitor ForesightFlow repo for implementation code (promised at github.com/ForesightFlow)
2. Incorporate ILS concepts into market screening methodology
3. Use deadline-ILS extension for deadline-based Polymarket contracts
