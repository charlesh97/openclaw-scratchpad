# The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book

**Source:** https://arxiv.org/abs/2604.24366

## What it does
A comprehensive microstructure study of Polymarket using 30 billion order book events across 52 days, joined to on-chain trade records. Reports eight stylized facts about Polymarket's market microstructure.

## Key Findings
1. **Longshot spread premium** — Wider spreads on extreme probabilities
2. **Depth profile closer to uniform** — Not just top-of-book concentration
3. **Null block-clock alignment** — No evidence of time-based manipulation
4. **Broad maker-wallet diversity** — Many unique liquidity providers
5. **Category-conditional spread differences** — Sports vs politics vs crypto spreads differ systematically
6. **Sub-50ms median ingestion delay** — But multi-second tail
7. **Self-counterparty wash share** — Median 1%, upper tail 22%

## Why it matters
This is the most granular analysis of the Polymarket order book available. Understanding these microstructure patterns is essential for:
- Positioning our arb detection thresholds correctly
- Understanding how quickly arb opportunities decay
- Estimating realistic fill probabilities and slippage

## Implementability: 3/5
Research paper — no code. But the empirical findings directly inform bot parameters (minimum spread, position sizing, category-specific tuning).

## Next Steps
1. Set minimum spread thresholds informed by the spread premium findings
2. Tune arb detection per market category (sports vs politics vs crypto)
3. Account for the wash trading tail in risk models
