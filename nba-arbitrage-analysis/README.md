# Arbitrage Analysis in Polymarket NBA Markets

**Source:** https://arxiv.org/abs/2605.00864  
**Recommendation:** MEDIUM  
**Published:** April 22, 2026  

## Summary

Empirical study of arbitrage opportunities in Polymarket NBA Moneyline and Spread markets. Finds that combinatorial arb across Moneyline-Spread pairs is more frequent (290 active episodes), concentrated in final minutes of live play. BUT 76.9% of opportunities are limited to just 14.8 shares average — severely liquidity-constrained.

## Key Findings

1. **Moneyline-Spread Combinatorial Arb** — 290 episodes, concentrated in final minutes
2. **Liquidity Ceiling** — 76.9% constrained to 14.8 shares average ($14.80 market value)
3. **Limits-to-Arbitrage** — confirms Shleifer & Vishny (1997): execution frictions prevent full correction even when mispricing is unambiguous
4. **Actionable at retail scale only** — opportunities exist but are structurally small

## Why It Matters

- Shows liquid sports markets still have exploitable arb
- But the size limit means it's only interesting for retail-scale accounts
- Validates the "liquidity shallowness" theory in decentralized prediction markets
- Directly applicable to NBA/non-crypto markets

## Risks

- Only NBA Moneyline and Spread studied — not generalizable to all sports
- Liquidity constraints mean no institutional deployment possible
- Live-play concentration means you need quick execution during game events
- Polymarket sports markets have thinner liquidity than crypto markets

## Implementability: 2/5

Interesting academically but limited practical value due to liquidity constraints. Only useful as a supplementary strategy for very small capital.

## Status: QUEUED
