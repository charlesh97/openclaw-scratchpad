# Summary: Unravelling the Probabilistic Forest

**arxiv.org/abs/2508.03474**

## One-Page TL;DR

Polymarket's binary-condition design creates structural arbitrage opportunities. The paper identifies two types:

1. **Market Rebalancing Arbitrage:** Within a single market, buy YES + NO when their sum < $1.00 and wait for resolution. Or in multi-condition markets, rebalance across outcome tokens to a risk-free position.

2. **Combinatorial Arbitrage:** Across related markets (e.g., "Biden wins popular vote" vs "Biden wins election"), when the implied joint probabilities violate basic probability axioms.

**Magnitude:** ~$40M extracted in one year.

**Key empirical findings:**
- Most arbitrage profits concentrate in high-volume crypto markets (BTC 15-min, ETH 1-hour)
- Simple YES+NO bundle arbitrage is now mostly captured by HFT bots
- Combinatorial arbitrage persists longer but requires more complex detection
- Arbitrage activity improves market efficiency (reduces pricing errors)

**For builders:** The paper provides the theoretical foundation and measurement methodology. Combinatorial arbitrage detection requires modeling event dependency graphs — the most promising frontier for algorithmic arb.
