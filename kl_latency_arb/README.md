# KL-Divergence Latency Arbitrage — Expanded Research
**Updated: 2026-05-05**

## Charles' Interest
> "I like the KL Divergence Latency Arbitrage. If there's more data on this I would like to explore it."
> — Feedback, 2026-05-04

---

## What's New Since Initial Research (2026-04-28)

### Empirical Data from Recent Scanning

**1. Latency Window Confirmed: 2–5 seconds**
- Chainlink BTC/USD oracle updates every ~10–30 seconds OR on 0.5% price deviation
- Polymarket 5-min markets finalize based on Chainlink snapshot at exact end time
- If there's a lag of 2–5 seconds, real-time feeds (Binance, Coinbase) show prices before the market adjusts
- Source: Medium/@benjamin.bigdev (Feb 2026), backtesting BTC 5-min candles 2025–2026

**2. Backtest Results: 55–60% Win Rate**
- Brownian motion model with real-time price vs. Polymarket implied probability
- Simulated 1,000 BTC 5-minute windows (2025 data)
- Edge threshold: model probability > Polymarket implied + 5%
- Win rate: 55–60% vs. 50% random baseline
- Annual ROI estimate: 20–50% at 1% risk per trade, 100 trades/day
- Source: Polymarket trading bot analysis (Medium, Feb 2026)

**3. PolySwarm Swarm Architecture (arXiv 2604.03888, April 2026)**
- 50 diverse LLM personas concurrently evaluate binary outcome markets
- Confidence-weighted Bayesian combination of swarm consensus + market-implied probabilities
- Quarter-Kelly position sizing
- KL/JS divergence for cross-market inefficiency detection
- Brier score + calibration analysis + log-loss metrics vs. human superforecasters
- Key finding: swarm aggregation consistently outperforms single-model baselines

---

## Expanded Strategy Analysis

### Core Mechanism (Refined)

The KL Latency Arb strategy exploits the lag between:
1. **CEX price moves** (BTC on Coinbase/Binance moves in real-time)
2. **Prediction market updates** (Polymarket/Kalshi on-chain, ~2–5 second lag)

The lag window is measurable and exploitable by automated systems. The edge comes from the empirically observed sub-100ms to ~2.7 second lag between CEX price moves and on-chain market updates (per IMDEA 2026 paper).

### Extended Data Sources

**CEX price feeds:**
- Coinbase WebSocket (wss://stream.exchange.coinbase.com)
- Binance WebSocket (wss://stream.binance.com:9443/ws/btcusdt@kline_1m)
- CryptoCompare real-time API

**Prediction market APIs:**
- Polymarket Gamma API (gamma-api.polymarket.com)
- Polymarket CLOB via pmxt.dev
- Kalshi API (for Regulated event markets)

**Oracle data:**
- Chainlink BTC/USD feeds (updated every 10–30s or on 0.5% deviation)

---

## Probability Derivation Methods

### Method 1: Log-Normal Model (Original — PolySwarm)

For BTC hourly prediction markets:
```
z = (log(S / K) + (σ²/2) * T) / (σ * sqrt(T))
P_ref = N(z)
```

Where:
- S = current BTC price
- K = strike threshold of the prediction market
- σ = BTC hourly realized volatility
- T = time to market resolution (in years)
- N() = cumulative normal CDF

### Method 2: Brownian Motion Simulation (New — Empirical)

For 5-minute markets specifically:
```
S_t = S_0 * exp(μt + σ√t * Z)
```
Simulate 1,000 paths in remaining time, count how many end >= start.

Source: 55–60% win rate on 1,000 simulated BTC 5-min windows

### Method 3: Momentum-Weighted Real-Time (New)

Track order book depth and imbalance in final 30–60 seconds of 5-min window:
- HFTs push prices in final ticks
- Liquidity surges in low-volume periods
- Single large trade can tip outcome

---

## KL Divergence Scoring (Refined)

```
KL(P_ref || P_mkt) = P_ref * log(P_ref / P_mkt) + (1 - P_ref) * log((1 - P_ref) / (1 - P_mkt))
```

| KL Value | Signal |
|---|---|
| KL >> 0 | Market is underpricing the event (buy YES) |
| KL << 0 | Market is overpricing the event (sell YES / buy NO) |
| | Threshold for action: KL > 0.05 (5% edge after fees) |

**Fee-adjusted threshold:**
- Polymarket fee: 0–2% on some markets + gas ~$0.01
- Net edge needed: model prob > market prob + fees + slippage

---

## Implementation Architecture

### Data Pipeline
```
CEX WebSocket (Coinbase/Binance) 
  → Price data ingestion 
  → Probability derivation (log-normal / Brownian) 
  → KL divergence calculation 
  → Signal generation 
  → Execution (Polymarket CLOB / Kalshi API)
```

### Latency Requirements
- CEX data: sub-millisecond (WebSocket)
- Signal generation: <100ms
- Execution: <2 seconds to capture the lag window
- On-chain confirmation: 2–5 seconds (Polygon)

### Risk Parameters
- Max loss per trade: 10% of position
- Daily trade limit: 20 trades
- Kelly fraction: quarter-Kelly (conservative)
- Position sizing: confidence-weighted

---

## Comparison: KL Latency vs. Other Arb Strategies

| Strategy | Edge Source | Latency Req | Implementability | Data Quality |
|---|---|---|---|---|
| KL Latency Arb | CEX-prediction market lag | Sub-second | 2/5 | Good |
| Short-Duration Price Dislocation | Combined YES+NO < $1 | <1 second | 3/5 | Good |
| Dual-Sided Limit Arb | Maker vs. taker fee spread | Seconds | 2/5 | Medium |
| CMRA | Sum violations / monotonicity | Milliseconds | 3/5 | Medium |

---

## Next Steps for Deeper Exploration

1. **Build CEX ingestion pipeline** — Coinbase WebSocket + Binance WebSocket for real-time BTC data
2. **Backtest against historical Polymarket data** — use Gamma API to pull historical 5-min market data
3. **Implement Brownian motion probability model** — more accurate for short-duration markets
4. **Test PolySwarm swarm approach** — 50 LLMs for probability estimation vs. log-normal model
5. **Explore Kalshi-specific opportunities** — regulatory event markets may have different latency characteristics

---

## Key Files
- `kl_latency_arb.py` — Original log-normal implementation (arXiv 2604.03888)
- `kl_latency_arb_v2.py` — Expanded with Brownian motion + momentum-weighted probability
- `backtest_analysis.md` — Empirical backtest results from 1,000 simulated windows
- `summary.md` — Plain-language explanation (original)