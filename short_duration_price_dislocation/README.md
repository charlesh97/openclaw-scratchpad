# Short-Duration Price Dislocation Arbitrage

## What It Is

Short-duration binary prediction markets (5–15 minute expiry) on Polymarket trade crypto price direction contracts (Up/Down). Because these markets resolve against an oracle feed and the order book is thin, the **combined price of YES + NO** occasionally dips below $1.00 — a risk-free arbitrage within a single platform. When that happens, buying both sides locks in a guaranteed profit at settlement.

**The Core Mechanism:**

```
edge = 1.00 - (price_YES + price_NO)
```

If `edge > 0` (after fees), both legs win at expiry regardless of outcome.

---

## How It Differs From arb-bot-main

arb-bot-main targets cross-platform price mismatches (Polymarket vs Kalshi). This strategy exploits **within-platform** dislocations in ultra-short-duration contracts, primarily on Polymarket. The opportunity class is distinct: it requires high-frequency scanning and fast execution, but the arb is cleaner because both legs clear on the same oracle.

Key differences:
- **Venue**: Polymarket-only (short-duration Up/Down markets)
- **Legs**: Both BUY orders on same platform (YES + NO), vs cross-platform arb
- **Speed**: Sub-second detection required; windows last milliseconds
- **Capital**: Smaller per-trade sizing ($1K–$5K) due to thin order books ($5K–$15K depth)
- **Edge type**: Oracle lag + thin book volatility, not cross-platform disagreement

---

## Implementability: 3/5

- Requires WebSocket streaming (not polling) for real-time order book
- Small order sizing needed (~$1K/round-trip to avoid slippage)
- Fee-aware edge calculation is critical (Polymarket taker fees 0.75–1.80%)
- Infrastructure: low-latency VPS near exchange endpoints
- Moderate coding complexity; the detection math is simple, execution is the hard part

---

## Recommendation: YES

The strategy is theoretically sound, documented in live trading (~$150K P&L reported across ~8,894 trades per Coindesk), and has a clean mathematical edge when the combined price falls below $1. The main risk is execution speed — the windows are millisecond-scale. For a bot with solid infrastructure, this is a genuine edge.

---

## Key Risks

1. **Execution risk**: Both legs must fill before the dislocation closes
2. **Slippage**: Order books are $5K–$15K deep per side; large orders move price
3. **Fee compression**: Taker fees (0.75–1.80%) can eliminate the edge entirely
4. **Oracle risk**: Settlement depends on Chainlink or equivalent oracle; disputes possible
5. **Infrastructure**: Requires co-location or low-latency setup most individuals can't match

---

## Sources

- **Medium / @gwrx2005 (March 2026)**: [AI-Augmented Arbitrage in Short-Duration Prediction Markets](https://medium.com/@gwrx2005/ai-augmented-arbitrage-in-short-duration-prediction-markets-live-trading-analysis-of-polymarkets-8ce1b8c5f362) — Live trading post-mortem, v2/v3 signal engine design
- **Coindesk (Feb 2026)**: [How AI is helping retail traders exploit prediction market 'glitches'](https://www.coindesk.com/markets/2026/02/21/how-ai-is-helping-retail-traders-exploit-prediction-market-glitches-to-make-easy-money) — ~$150K P&L reported, 8,894 trades
- **IndieHackers (2026)**: [Latency Arbitrage in 15-Minute Crypto Markets](https://www.indiehackers.com/post/latency-arbitrage-in-15-minute-crypto-markets-building-a-polymarket-trading-edge-2026-f77cc226c0) — Framework for latency-based reaction strategies
- **arXiv:2604.03888** (PolySwarm): Already covered in `kl_latency_arb/` — KL divergence reference model for crypto price feeds

---

## Next Steps

1. Build WebSocket stream for Polymarket order book (use `polymarket.com/api` or Gamma API)
2. Implement price_sum < 1.00 detector with rolling window (detect within 100ms)
3. Fee-aware edge calculation: `edge = 1.00 - (yes_price + no_price) - fees`
4. Maker-order preference to reduce fee load; only use taker when edge > threshold
5. Position sizing capped at $2K per round-trip to avoid book impact
6. Early exit logic: take profit at 0.80–0.95 rather than holding to settlement
