# Summary: Anatomy of a Decentralized Prediction Market

**arXiv:2604.24366** — Dubach et al., April 2026

## TL;DR

The first comprehensive microstructure study of Polymarket using 30B order-book events across 600 markets. Eight key stylized facts about spreads, depth, latency, and wash trading. Critical warning: order-book feed trade direction is only ~59% accurate — use on-chain data.

## Key Numbers

| Metric | Value |
|--------|-------|
| Data volume | 30B events over 52 days |
| Market panel | 600 stratified markets |
| Median ingestion delay | <50ms |
| Wash trade median | 1% (22% tail) |
| Feed vs on-chain trade direction agreement | ~59% |
| Depth decay slope (log seconds-to-close) | 0.55 (t=3.85) |

## For Bot Builders

- **Latency**: Median trade ingestion is fast (<50ms) but tail is multi-second — don't assume uniform latency
- **Depth**: Much more uniform across price levels than equity markets — affects optimal order placement
- **Spreads**: Vary significantly by category (crypto markets tighter than sports)
- **Wash trading**: Low enough to not be a primary concern, but 22% tail means some markets are compromised
- **Data caveat**: Always use on-chain OrderFilled events for trade direction analysis
