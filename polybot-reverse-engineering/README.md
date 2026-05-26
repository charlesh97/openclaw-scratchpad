# Polybot — Polymarket Strategy Reverse-Engineering Toolkit

**Source:** https://github.com/ent0n29/polybot  
**Author:** ent0n29  
**Language:** Java 21 (microservices) + Python (research)  
**License:** MIT  
**Status:** Active

## What It Does

An open-source Polymarket trading infrastructure and strategy reverse-engineering toolkit. Polybot is a multi-service system designed to ingest all Polymarket market/user trade data, run strategy backtests, and score replication accuracy. It's the execution and data foundation for **AWARE** (trader intelligence, PSI indices, fund mirroring).

### Key Components

- **Executor Service** — Automated execution (paper and live)
- **Strategy Service** — Strategy runtime and market making
- **Ingestor Service** — Market/user trade ingestion into ClickHouse
- **Analytics Service** — Quantitative analysis and replication scoring
- **Research Toolkit** — Python scripts for snapshot analysis, deep analysis, and replication metrics

### Tech Stack
- Java 21 Spring Boot microservices
- ClickHouse (analytics database)
- Redpanda (event streaming)
- Grafana + Prometheus (monitoring)
- Python 3.11+ (research scripts)

## Why It Matters

- **Strategy reverse-engineering** — Unique capability to analyze what strategies top traders are using
- **Research-grade infrastructure** — ClickHouse + Redpanda for fast analytics on millions of trades
- **AWARE integration** — PSI (Polymarket Sentiment Index) and fund mirroring on roadmap
- **Enterprise-grade** — Java microservices with proper monitoring, not a weekend Python script

## Risks

- Heavy infrastructure requirements (ClickHouse, Redpanda, multiple microservices)
- Java stack — different from our Python-focused toolchain
- More of a research/analytics platform than a trading bot
- Requires significant RAM for ClickHouse

## Implementability: 2/5

- Complex setup requiring Docker, Java 21, Maven, and ClickHouse
- Better suited as an intelligence/research layer than as a trading engine
- Worth integrating the research Python scripts into our workflow

## Next Steps

1. Review the research Python scripts for trade replication analysis
2. Consider ClickHouse as our analytics backend if scale grows
3. Monitor AWARE product layer for PSI indices
