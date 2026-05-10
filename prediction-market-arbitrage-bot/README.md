# prediction-market-arbitrage-bot

**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot
**Type:** Cross-platform arbitrage bot
**Recommendation:** YES

## What It Does

Detects and executes arbitrage strategies between Polymarket and Kalshi for the same prediction event. The core mechanism is simple and powerful:

1. **Same-event price comparison** — Compare Polymarket vs. Kalshi prices for the same outcome
2. **Auto-buy low / sell high** — When Polymarket price < Kalshi price for YES, buy Polymarket, sell Kalshi (and vice versa)
3. **Built on pmxt.dev** — Uses the pmxt.dev infrastructure for market data and execution

## Why It Matters

Cross-platform arbitrage between Polymarket and Kalshi is one of the most accessible arb strategies because:
- Both platforms list many of the same events (US elections, economic data, sports)
- Price discrepancies are visible and quantifiable
- No complex combinatorial modeling required (unlike inter-market combinatorial arb)

## Key Considerations

- **Capital requirements** — Need funds on both platforms to execute simultaneously
- **Kalshi integration** — Requires separate API credentials and account setup for each platform
- **Execution latency** — Price discrepancy must survive the round-trip; need ~100-200ms max latency
- **Withdrawal/funding delays** — Moving USDC on Polymarket vs. USD on Kalshi introduces friction

## Implementability: 4/5

Clean, focused implementation of cross-platform arb. Lower complexity than ImMike's multi-mode system — better starting point if you want to understand the cross-platform flow before adding market-making or bundle detection layers.

**Next steps:**
1. Set up pmxt.dev account and API access
2. Clone and run the bot with paper-trade mode
3. Add your own price discrepancy threshold tuning
4. Add position tracking and P&L monitoring