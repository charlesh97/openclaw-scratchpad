# KL-Divergence Latency Arbitrage

## What It Is

A prediction market arbitrage strategy that derives a **reference probability** from external CEX (centralized exchange) data, then exploits the time lag between when CEX prices move and when Polymarket/Kalshi prices update.

**Core mechanism (from PolySwarm, arXiv 2604.03888):**
1. Derive an implied probability from CEX prices using a log-normal pricing model (e.g., BTC hourly close prediction → Black-Scholes z-score → probability)
2. Compare reference probability vs. Polymarket/Kalshi market price using **Kullback-Leibler (KL) divergence** as a scoring signal
3. When KL divergence exceeds a threshold, bet against the Polymarket price in the direction of the reference probability
4. The "edge" comes from the empirically observed sub-100ms to ~2.7 second lag between CEX price moves and on-chain market updates

**Why it works:**
- Prediction markets on BTC/S&P outcomes track their underlying CEX reference
- Large CEX moves (e.g., BTC dumps 2% in an hour) cause prediction markets to update, but with a lag
- The lag window is measurable and exploitable by automated systems
- KL divergence quantifies how much "surprise" the market price represents relative to the reference probability

## How It Differs From arb-bot-main

arb-bot-main's `check_time_decay()` uses time-to-expiry to estimate fair value. This approach uses **external CEX data as the ground-truth reference**, measuring the market's deviation from a derived probability rather than from a time-decay model. It's complementary: use time-decay for long-dated/earnings events, use KL-latency for real-time/volatile events.

## KL Divergence Formula

Given reference probability `P_ref` and market probability `P_mkt`:

```
KL(P_ref || P_mkt) = P_ref * log(P_ref / P_mkt) + (1 - P_ref) * log((1 - P_ref) / (1 - P_mkt))
```

When KL >> 0: market is underpricing the event (relative to reference)
When KL << 0: market is overpricing the event

## Implementation

### BTC Probability Derivation (Log-Normal Model)

For BTC hourly prediction markets, derive probability from BTC volatility:

```
z = (log(S / K) + (σ²/2) * T) / (σ * sqrt(T))
P_ref = N(z)  # cumulative normal
```

Where:
- S = current BTC price
- K = strike threshold of the prediction market
- σ = BTC hourly realized volatility
- T = time to market resolution (in years)
- N() = cumulative normal CDF

### Data Sources

- BTC real-time price: Coinbase, Binance, or CryptoCompare API
- BTC hourly volatility: derived from recent price history
- Polymarket CLOB: via pmxt.dev or direct API
- Kalshi: via Kalshi API

### Feasibility: 2/5

Requires reliable real-time CEX data ingestion + on-chain price feeds + sub-second execution. The empirical opportunity window is ~2.7 seconds on average (per IMDEA 2026 data), which demands low-latency infrastructure. Not a pure detection algorithm — requires execution infrastructure.

## Risks

- **Latency risk**: opportunity window is 2.7s on average; sub-second execution required
- **Model risk**: log-normal assumption may not hold for all market conditions
- **Liquidity risk**: Polymarket CLOB may not have enough depth to absorb the arb size
- **Regulatory risk**: arbitrage at this frequency may attract attention

## Next Steps

1. Build CEX price ingestion pipeline (Coinbase WebSocket or CryptoCompare)
2. Implement z-score → probability derivation for BTC and S&P500 markets
3. Backtest KL signal against historical Polymarket data
4. Build execution layer for Polymarket CLOB (pmxt.dev or direct API)
