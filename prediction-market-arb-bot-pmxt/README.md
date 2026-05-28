# Prediction Market Arbitrage Bot (pmxt.dev)

**Source:** [github.com/realfishsam/prediction-market-arbitrage-bot](https://github.com/realfishsam/prediction-market-arbitrage-bot)

## What It Does

A **cross-platform arbitrage bot** that detects and automatically executes arbitrage strategies between Polymarket and Kalshi. Built on top of the [pmxt.dev](https://pmxt.dev) cross-platform market data API.

## Key Features

- **Cross-platform arb detection** — monitors Polymarket + Kalshi for the same events
- **Auto-execution** — buys low on one platform, sells high on the other
- **pmxt.dev integration** — uses a dedicated cross-platform data API for market matching
- **Configurable edge thresholds** — set minimum profit % to filter noise
- **Dry run mode** — test before real deployment

## Why It Matters

Cross-platform arbitrage between Polymarket and Kalshi is one of the most consistently profitable strategies in prediction markets:
- Same event, different user bases create price dislocations
- Kalshi's CFTC-regulated environment vs. Polymarket's unregulated DeFi creates different trader populations
- pmxt.dev provides a dedicated API for matching markets across platforms (removing the hardest part — market discovery)

## Risks

- **Execution risk:** By the time you arb, the opportunity may be gone
- **Kalshi API limits:** CFTC-regulated platform may have additional constraints
- **Withdrawal timing:** Funds may be locked during settlement
- **Gas costs:** Polymarket tx fees eat into thin edges
- **Counterparty risk:** pmxt.dev is a third-party dependency

## Implementability: 4/5

Well-structured Python codebase with clear strategy logic. The pmxt.dev integration handles the hardest part (cross-platform market matching).

## Next Steps

1. Set up pmxt.dev API access
2. Deploy in dry run mode to validate opportunity frequency
3. Test with minimal capital ($50-100) to measure real execution quality
4. Compare against ImMike's existing cross-platform implementation
5. Evaluate merging both approaches for broader coverage
