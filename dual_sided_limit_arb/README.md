# Dual-Sided Limit Arbitrage with Maker Optimization

## What It Is

Rather than paying taker fees to immediately fill both sides of an arbitrage, this strategy places **maker limit orders on BOTH sides** of a binary market simultaneously. The trader earns the maker rebate (effectively "getting paid" to provide liquidity) while waiting for the market to converge. The dual-maker approach turns the fee structure into an edge rather than a cost.

**Core Logic:**

```
If P(YES) + P(NO) < 1 - maker_fee_both_legs  (maker rebate means we earn, not pay)
  → Place: BUY YES @ bid  +  BUY NO @ bid
  → Earn rebate on both legs
  → Wait for convergence or settlement
```

With Polymarket's ~0% maker rebate and ~1.75% taker fee (as of 2026), pure taker strategies are squeezed. Dual-maker flips the math.

---

## How It Differs From arb-bot-main

arb-bot-main uses aggressive market orders ("YOLO mode") to instantly capture spreads. That approach works when:
- Edge >> fees (rare)
- Speed > position quality

This strategy takes the opposite approach:
- **Patience over speed**: earn the spread by posting both sides
- **Fee inversion**: maker rebates make liquidity provision profitable even in tight markets
- **Reduced book impact**: limit orders don't move the order book

The two strategies are complementary: use market orders when edge is large, use dual-maker when edge is small but durable.

---

## Implementability: 2/5

- Requires CLOB access with maker order capabilities
- Needs position tracking: total YES + NO exposure must be balanced
- Works best in trending or volatile markets (wider spreads = more edge)
- Infrastructure: WebSocket for order acknowledgment and fill confirmations
- Moderate complexity: the math is simple, the operational challenge is managing dual-leg risk

---

## Recommendation: MEDIUM (conditional)

**Case FOR:**
- Fee inversion is real: maker vs taker cost difference is structural, not incidental
- Works well in Polymarket's 5-min Up/Down markets where spreads are wide (2–4 cents)
- Dual-maker is low-risk if both legs are filled (market-neutral at all times)

**Case AGAINST:**
- Polymarket maker rebate is currently ~0%; the advantage is mainly avoiding the taker fee
- If only one leg fills, you hold a directional position — requires active monitoring
- Spread compression as markets converge can eliminate maker edge before fills occur
- Needs real CLOB integration (not available via simple public API)

---

## Key Risks

1. **Partial fill risk**: If only one leg fills, you're exposed directional — must hedge or exit
2. **Spread compression**: Markets converge fast in short-duration contracts; maker quotes may get undercut
3. **Opportunity cost**: Capital tied up in open maker orders unavailable for other trades
4. **Cross-platform execution**: Kalshi has different maker fee structure; cannot directly dual-maker across venues
5. **Rebate risk**: If Polymarket eliminates maker rebates, the strategy degrades to taker-level economics

---

## Sources

- **DeepWiki (2026)**: [Maker vs Taker Strategies — dev-protocol/polymarket-ai-synth-trading](https://deepwiki.com/dev-protocol/polymarket-ai-synth-trading-bot-telegram/9.2-maker-vs-taker-strategies) — Technical breakdown of Polymarket CLOB fee structure post-Feb 2026 rule changes
- **AhaSignals (2026)**: [Polymarket vs Kalshi Arbitrage 2026](https://ahasignals.com/research/prediction-market-arbitrage-strategies/) — Fee-adjusted spread methodology
- **Coindesk (Feb 2026)**: [How AI is helping retail traders exploit prediction market 'glitches'](https://www.coindesk.com/markets/2026/02/21/how-ai-is-helping-retail-traders-exploit-prediction-market-glitches-to-make-easy-money) — Order book depth context ($5K–$15K per side)
- **pmxt.dev**: Unified prediction market API — includes Polymarket CLOB endpoints

---

## Next Steps

1. Integrate with Polymarket CLOB via `pmxt` API or Gamma API for maker order placement
2. Implement dual-leg fill tracking: both legs must fill within a configurable time window
3. If partial fill (one leg only): auto-hedge on the opposite platform or close at market
4. Build a "maker quality" score: estimate probability of fill based on spread width, book depth
5. Backtest against historical 5-min BTC market data to measure fill rate vs. edge tradeoff
6. Consider resting orders slightly inside the spread (0.2–0.5 cents) to increase fill probability while still profiting
