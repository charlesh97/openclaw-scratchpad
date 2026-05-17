# Polybot & AWARE (ent0n29)

**Source:** https://github.com/ent0n29/polybot | https://github.com/ent0n29/aware

## What It Does
Open-source Polymarket trading infrastructure and strategy reverse-engineering toolkit. Polybot is a multi-service Java microservices system for automated execution, strategy runtime, market making, and quantitative analysis. AWARE is the next layer: trader intelligence, PSI indices, fund mirroring, and API/UI.

## Why It Matters
- Full event pipeline: ClickHouse + Redpanda for trade ingestion
- Strategy reverse-engineering: replicates and scores other traders' strategies
- Production-grade: Java 21 microservices, Docker Compose, Grafana stack
- Strategy runtime supports market making, arbitrage, and custom strategies
- AWARE adds PSI (Polymarket Sentiment Index) and fund mirroring

## Implementability: 4/5
**MEDIUM** — enterprise-grade stack (Java, ClickHouse, Redpanda) is powerful but has a steep learning curve for non-Java teams. The strategy replication scoring is genuinely novel.

## Next Steps
1. Deploy the ingestion pipeline to capture live Polymarket trade data
2. Run strategy replication on top 10 Polymarket wallets
3. Export strategy parameters for integration with simpler execution engine
