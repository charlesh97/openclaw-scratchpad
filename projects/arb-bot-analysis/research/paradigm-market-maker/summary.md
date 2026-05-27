# Summary: Anatomy of a Decentralized Prediction Market

**arXiv:2604.24366** | Apr 2026

## One-Page TL;DR

The most comprehensive microstructure study of Polymarket to date: **30 billion order-book events** over 52 days across **600 markets**. Key findings:

### 8 Stylized Facts
1. **Longshot spread premium** — less liquid outcomes have wider spreads
2. **Uniform depth** — more uniform than top-of-book pattern assumed for prediction markets
3. **No block-clock alignment** — block times don't affect liquidity patterns
4. **Broad maker diversity** — many wallets, but top wallets dominate
5. **Category-conditional spreads** — sports vs crypto vs politics differ significantly
6. **Sub-50ms ingestion delay** — median latency, but multi-second tail
7. **Wash trading** — median 1% (low), 22% upper tail (moderate concern)
8. **Cross-sectional depth** — driven by duration, price level, volume

### Critical Measurement Result
Trade direction inferred from public order-book feed agrees with on-chain truth only **~59% of the time** (vs ~80% on Nasdaq). Microstructure work must source trade direction from on-chain `OrderFilled` events.

**Replication package:** [github.com/philippdubach/polymarket-microstructure](https://github.com/philippdubach/polymarket-microstructure)
