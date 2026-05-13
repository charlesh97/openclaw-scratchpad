# Cross-Platform Arbitrage Bot (Polymarket ↔ Kalshi)

**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot

## What it does

A bot that detects and executes arbitrage strategies between **Polymarket** and **Kalshi** — the two largest prediction markets. Built with pmxt.dev, it auto-buys low and sells high across platforms when price discrepancies exist for the same event.

## Key features

- **Cross-platform price comparison:** Matches identical events on both platforms
- **Automated execution:** Places orders on both sides simultaneously
- **fee-aware calculations:** Accounts for platform fees when computing arbitrage profit
- **Real-time monitoring:** Continuous scanning for opportunities

## Why it matters

Cross-platform arb is one of the most reliable strategies because prices *must* converge to $1.00 at resolution on both platforms. Any persistent price gap is a genuine arbitrage opportunity. Polymarket removing the 500ms taker delay in early 2026 makes execution more competitive but also more reliable.

## Implementability: 4/5

Well-documented bot with pmxt.dev integration. Requires API keys for both platforms. PMXT abstracts most of the complexity.

## Risks

- Both platforms may not list identical markets
- Withdrawal/settlement timing differences across platforms
- Fee structures differ — must compute net profit carefully
- Latency advantage matters — dedicated RPC needed

## Next Steps

1. Set up Polymarket API credentials
2. Set up Kalshi API credentials (requires KYC)
3. Deploy pmxt.dev bridge
4. Run paper trading for 1 week to validate spread detection
