# KL Divergence Latency Arbitrage — Plain Summary

**Source:** PolySwarm: A Multi-Agent LLM Framework for Prediction Market Trading and Latency Arbitrage
**arXiv:** 2604.03888 | April 2026 | Barot & Borkhatariya

---

## What Is This?

A way to automatically find mispriced prediction markets by comparing what the market thinks will happen vs. what external data (like crypto prices) says should happen. When there's a gap, there's potential profit.

---

## How It Works (Simple Version)

1. **Get real-time CEX prices** — e.g., current BTC price from Coinbase or Binance
2. **Convert to probability** — use a math formula (log-normal model) to turn "BTC is at $104,500" into a probability that BTC will be above $105,000 in an hour
3. **Compare to Polymarket price** — if Polymarket says 52% but the math says 55%, that's a gap
4. **Measure the gap with KL divergence** — a number that tells you how "surprised" the market price is relative to the CEX price
5. **Act when the gap is big enough** — buy YES if the market underestimates, sell YES if it overestimates

---

## Why It Makes Money

Prediction markets track real-world events. When Bitcoin moves 2% on Coinbase, prediction markets on "Will BTC be above $X?" should move too — but they move slowly (about 2.7 seconds on average). Algorithmic traders exploit this lag.

---

## What "KL Divergence" Means

KL divergence is a way to measure the difference between two probabilities. Think of it like a "surprise score": if the prediction market says 60% but our model says 40%, the KL score tells us how surprised we should be. Bigger surprise = bigger potential edge.

---

## Quarter-Kelly Sizing

Once you find an edge, how much do you bet? **Quarter-Kelly** is a conservative sizing formula that grows your bankroll safely while avoiding overbetting. It says: bet a fraction of your bankroll proportional to your edge, but never more than 25% of your total.

---

## What Could Go Wrong

- **Speed:** The opportunity window is ~2.7 seconds. You need fast execution.
- **Model risk:** The math formula assumes prices move randomly (lognormal). This isn't always true.
- **Liquidity:** The Polymarket order book might not have enough depth to absorb large bets.
- **Fee drag:** Fees on both sides can eat your edge.
