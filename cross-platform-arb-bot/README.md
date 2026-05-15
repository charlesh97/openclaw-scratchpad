# Cross-Platform Arbitrage Bot (Polymarket ↔ Kalshi)

**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot  
**Recommendation:** YES

## What It Does

An educational bot that detects and executes synthetic arbitrage strategies between Polymarket and Kalshi. Implements "synthetic arbitrage": buying YES on one platform and NO on the other for the same outcome, locking in profit when prices converge. Uses fuzzy matching (Jaccard + Levenshtein distance) to pair outcomes across platforms.

## Example Arbitrage

- Polymarket: Kevin Warsh YES = 41¢
- Kalshi: Kevin Warsh NO = 57¢
- Total cost: 98¢ → Guaranteed 2¢ profit per share

## Key Features

- Cross-platform price scanning via pmxt.dev API
- Two arbitrage strategies (YES on Polymarket + NO on Kalshi, and vice versa)
- YOLO mode (all-in on best opportunity) or CONSERVATIVE mode
- Fuzzy matching handles naming variations across platforms
- Dry run mode for testing

## Why It Matters

Directly addresses the $40M+ arbitrage opportunity documented in the "Unravelling the Probabilistic Forest" paper. Cross-platform arbitrage is one of the most mechanically sound strategies because it exploits real price differences between two marketplaces for the same underlying event.

## Implementability: 4/5

Simple Node.js application, easy to configure. Requires API keys for both Polymarket and Kalshi. The pmxt.dev API handles unified access.

## Risks

- Requires Kalshi account (US only)
- Simultaneous execution is critical
- Liquidity on both platforms must be sufficient
- Regulatory risk on Kalshi side
