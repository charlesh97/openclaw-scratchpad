# prediction-market-arbitrage-bot (Polymarket–Kalshi)

**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot
**Recommendation:** YES (email candidate #2)

## What it does

Cross-platform arbitrage between **Polymarket** and **Kalshi** — the two deepest regulated/pseudo-regulated prediction markets in the US. When the same event (e.g., "Will BTC be above $X at time T?") is priced differently on both platforms, the bot:

1. Buys the cheaper side on Platform A
2. Sells the equivalent position on Platform B (or vice versa)
3. Captures the spread when both sides resolve

The bot watches 10,000+ markets simultaneously and uses `pmxt.dev` as the unified price feed. Built specifically for BTC price prediction contracts that exist on both venues.

## Why it matters

- True cross-exchange arbitrage is the most robust form — it exploits structural price discovery inefficiencies between venues, not just intra-market AMM spreads
- Kalshi is CFTC-regulated and has retail/US-restricted access; Polymarket is crypto-native — the regulatory gap creates persistent pricing discrepancies
- The fact that the same BTC 1-hour and 15-minute contracts exist on both platforms makes this a recurring, predictable strategy
- Published recently and actively maintained (Awesome-Prediction-Market-Tools listing, 1 week ago)

## Risks

- **Platform access risk:** Kalshi has US trading restrictions — not all users can arb from Kalshi directly. The bot likely uses Kalshi's public API which may have rate limits or be geofenced.
- **Execution latency:** Cross-platform settlement times differ; Polymarket resolves on-chain (Polygon), Kalshi resolves on USDC or fiat. Settlement lag creates execution risk.
- **Regulatory risk:** CFTC has signaled interest in Kalshi's market structure. If Kalshi changes terms, the arb collapses.
- **Position sizing:** Both platforms require margin/balance management — a single bad outcome on one side while the other is pending can create a directional bet rather than a true hedge.

## Implementability: 4/5

- Relatively straightforward architecture: monitor both APIs → detect spread → execute both legs
- `pmxt.dev` integration simplifies the dual price feed problem
- Requires accounts + API keys on both platforms
- Needs a state machine for tracking open positions across venues

## Next Steps

1. Get Kalshi API credentials (apply for developer access)
2. Set up `pmxt.dev` account for unified price feed
3. Backtest across 6 months of overlapping BTC contract data
4. Build position reconciliation system for cross-platform settlements

## Sample Logic (pseudocode)

```python
# Cross-platform arb between Polymarket and Kalshi
def check_cross_platform_arb(event_id):
    pm_price = pmxt.get_polymarket_price(event_id)
    kalshi_price = pmxt.get_kalshi_price(event_id)

    if pm_price < kalshi_price - 0.02:  # 2% spread threshold
        # Polymarket is cheaper — buy YES on Polymarket, sell equivalent on Kalshi
        size = calculate_hedge_size(pm_price, kalshi_price, balance)
        polymarket.buy(event_id, size, pm_price)
        kalshi.sell(event_id, size, kalshi_price)
        log_arb_opportunity(event_id, pm_price, kalshi_price, size)
    
    elif kalshi_price < pm_price - 0.02:
        # Kalshi is cheaper — mirror
        size = calculate_hedge_size(kalshi_price, pm_price, balance)
        kalshi.buy(event_id, size, kalshi_price)
        polymarket.sell(event_id, size, pm_price)
        log_arb_opportunity(event_id, pm_price, kalshi_price, size)
```

---
*Part of vega's arb-bot-analysis research.*