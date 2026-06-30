# 2026-06-30 Crypto-Specific Follow-up

## Market Size
- **$70M daily volume** on 5-min + 15-min crypto PM markets (Sahm Capital, Mar 2026)
- 5-min crypto bets now dominate prediction flows (crypto.news, Jun 2026)
- Volume still climbing after Polymarket added fees Jan 2026

## Core Inefficiency
- Spot prices move instantly (Binance/Coinbase WebSocket)
- PM probabilities update with delay (oracle latency, order book friction)
- Latency arb framework: Benjamin Cup, Apr 2026
  → https://benjamincup.substack.com/p/latency-arbitrage-in-15-minute-crypto

## Best Execution Engine
### PolyHFT (TheOverLordEA) ⭐58 — Rust
→ https://github.com/TheOverLordEA/polymarket-hft-engine
- <5ms tick-to-trade, AWS eu-west-1 colocated
- De-peg killswitch, velocity lockout, volatility desert, drift reconciliation
- Telegram alerts, institutional risk management
- btc-5min-bot + eth-5min-bot binaries
- Updated Jun 30, 2026 — active development

## Scratchpad Crypto Stack (Re-prioritized)

### Tier 1
| # | Resource | Lang | Stars | Notes |
|---|----------|------|-------|-------|
| 1 | PolyHFT | Rust | 58 | Production execution engine |
| 2 | Polymarket Dynamic Fees | — | — | Kills naive latency arb, enables maker strats |
| 3 | BTC 15-Min Bot (aulekator) | Python | — | 7-phase architecture, self-learning |

### Tier 2
| # | Resource | Lang | Notes |
|---|----------|------|-------|
| 4 | CarlosIbCu BTC-Kalshi | Python | Docker, thesis doc |
| 5 | WSOL12 BTC Scanner | Python | Auto-matching, FastAPI+Next.js |
| 6 | ImMike Cross-Platform | Python | 10K+ markets, 3 strategies |
| 7 | PolyBullLabs Short-Horizon | Python | 3 bots: 5m/15m/1h |
| 8 | Benjamin Cup Python bot | Python | Companion to latency arb article |

### Tier 3
| # | Resource | Notes |
|---|----------|-------|
| 9 | Trum3it Arb Bot | New, untested |
| 10 | Reddit r/algotrading 5-min thread | Community discussion |

## Academic
### PIRAP Perpetual Futures (arXiv:2605.10400)
→ https://arxiv.org/abs/2605.10400
- Crypto-perp design fails on PM underlyings (3 structural breaks)
- Bounded support, asymmetric depth, oracle resolution
- 4-component fix proposed, 3/5 materiality floors fail
- Code: github.com/ForesightFlow/event-linked-perps

### Polymarket Microstructure (arXiv:2604.24366)
- Category-conditional spreads: crypto ≠ sports ≠ politics
- Trade direction from order-book = 59% accurate → use on-chain
- Sub-50ms ingestion but multi-second tail

## Market Context
- Polymarket fees: ~3.15% at 50/50 on 15-min crypto (Jan 2026)
- Kalshi short-term crypto forwards: Dec 2025, ~50% of crypto flow
- Nasdaq filing binary options on Nasdaq 100 — TradFi copying PM

## Recommended Build Path
1. Clone PolyHFT → AWS eu-west-1 → dry-run 5-min BTC
2. Add simple latency arb strategy (EV > fee hurdle)
3. Cross-platform when spreads exceed single-platform fees
4. Extend BTC → ETH → SOL, 5-min → 15-min
