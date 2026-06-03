# Polybot — Polymarket Strategy Infrastructure

**Source:** https://github.com/ent0n29/polybot

## What It Does

Multi-service Java 21 microservices platform for Polymarket:
- **Executor Service** — Order execution (paper + live)
- **Strategy Service** — Strategy runtime and market making
- **Ingestion Service** — Market/user trade ingestion into ClickHouse
- **Analytics Service** — Quantitative analysis and replication scoring
- **Infrastructure Orchestrator** — Coordinates infrastructure stack

Built on Java 21, Spring Boot, ClickHouse (columnar DB), and Redpanda (Kafka-compatible event streaming).

Also has a companion project: **AWARE** (https://github.com/ent0n29/aware) — trader intelligence, PSI indices, fund mirroring, API/UI.

## Why It Matters

This is the most **enterprise-grade** Polymarket infrastructure toolkit available. The approach — reverse-engineering every strategy and quantifying replication scoring — is uniquely valuable. If you want to understand what strategies profitable traders are actually running, this is the infrastructure to figure it out.

## Risks
- Heavy infrastructure: Java 21, ClickHouse, Redpanda, Spring Boot — significant operational overhead
- No documentation on strategy models themselves (it's an infrastructure toolkit)
- Requires real Polymarket API keys
- Spring Boot microservices are complex to tune and debug

## Implementability: 2/5

Serious operational overhead. Best suited as a reference architecture rather than something to deploy immediately. The ingestion pipeline and ClickHouse schema are worth studying.

## Next Steps
1. Study the ClickHouse schema for market data modeling
2. Review the ingestion pipeline architecture
3. Consider extracting just the ingestion + ClickHouse components for lighter deployment
