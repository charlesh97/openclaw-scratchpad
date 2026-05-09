# polymarket-arbitrage-bot (Dump Hedge Strategy)

**Source:** https://github.com/dev-protocol/polymarket-arbitrage-bot
**Published:** March 16, 2026
**Recommendation:** YES (top pick today)

## What it does

The bot monitors UP/DOWN token price movements on Polymarket's short-duration BTC price prediction markets (15-minute and 1-hour horizons). When the spread between the two outcome tokens deviates from $1.00 parity — a structural arbitrage window — it executes a "dump hedge" strategy:

- If the YES (UP) token drifts below its fair value and the NO (DOWN) token rises, the bot buys YES and NO simultaneously, locking in a guaranteed spread capture when both resolve.
- Conversely, when YES is overvalued vs. NO, it can short or restructure the hedge.

**Core mechanic:** Polymarket's AMM model means YES + NO don't always sum to $1.00 due to liquidity asymmetry. The bot detects this delta and deploys capital to capture the mispricing before the market self-corrects.

## Why it matters

- Active strategy, not just theoretical — published March 2026
- Targets the same BTC price markets that `aulekator/Polymarket-BTC-15-Minute-Trading-Bot` focuses on but adds a specific arbitrage leg that pure directional bots miss
- Works on the same short-duration contracts that have seen the most rapid mispricing post-news events

## Risks

- Execution latency is critical — HFT bots will capture these spreads in milliseconds; a retail bot needs RPC edge + careful sizing
- Shallow order books cap maximum position size (~14.8 shares per episode per NBA arbitrage paper, same dynamic applies here)
- Smart contract execution risk on Polygon; gas spikes can erode thin arbitrage margins
- No explicit stop-loss described — single-side position risk if the market resolves differently than expected

## Implementability: 4/5

- Well-structured GitHub repo with clear strategy description
- Language/framework not specified in search results — needs code review
- Requires running a Polygon RPC node for low-latency order book monitoring
- Can be adapted to Python or Rust with the `pmxt.dev` API layer

## Next Steps

1. Review actual source code for implementation language and dependencies
2. Set up Polymarket API credentials and Polygon RPC access
3. Paper-trade the hedge ratio logic against historical order book snapshots
4. Integrate with `pmxt.dev` for unified cross-platform price feeds

## Sample Logic (pseudocode)

```python
# Dump Hedge — detect mispricing between YES and NO tokens
def check_arbitrage(market_id):
    yes_price = get_yes_price(market_id)
    no_price = get_no_price(market_id)
    spread = yes_price + no_price  # should = 1.00

    if spread < 0.98:
        # YES underpriced relative to NO — buy YES, sell NO equivalent
        size = min(balance * 0.5, max_position)
        execute_buy("YES", market_id, size)
        execute_sell("NO", market_id, size * (yes_price / no_price))
        log_arbitrage(spread, size)
    elif spread > 1.02:
        # NO underpriced — mirror position
        size = min(balance * 0.5, max_position)
        execute_buy("NO", market_id, size)
        execute_sell("YES", market_id, size * (no_price / yes_price))
        log_arbitrage(spread, size)
```

---
*Part of vega's arb-bot-analysis research. Update the algorithms table in /projects/arb-bot-analysis/research/README.md if adding to the canonical list.*