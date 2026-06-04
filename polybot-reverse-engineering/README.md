# PolyBot — Strategy Reverse-Engineering Toolkit

**Source:** https://github.com/ent0n29/polybot  
**Recommendation:** MEDIUM  
**Language:** Java 21 microservices + ClickHouse + Redpanda  

## What It Does

An enterprise-grade open-source Polymarket trading infrastructure and strategy reverse-engineering toolkit. Polybot is a multi-service system for automated execution, strategy runtime, market-making, user trade ingestion into ClickHouse, and quantitative analysis/replication scoring.

## Architecture (5 Microservices)

| Service | Port | Purpose |
|---------|------|---------|
| executor-service | 8080 | Order execution, paper sim, settlement |
| strategy-service | 8081 | Strategy runtime and status |
| analytics-service | 8082 | Analytics on ClickHouse data |
| ingestor-service | 8083 | Market/user-trade ingestion pipelines |
| infra-orchestrator | 8084 | Lifecycle of analytics + monitoring |

Infrastructure: ClickHouse, Redpanda Kafka, Grafana, Prometheus, Alertmanager

## Why It Matters

- **Reverse-engineering focus** — designed to replicate successful Polymarket strategies
- **Industrial stack** — Java 21 + ClickHouse is serious infrastructure
- **AWARE product layer** — future API/UI layer for trader intelligence
- **Complete monitoring** — full observability stack

## Risks

- High infrastructure complexity (Docker, ClickHouse, Kafka, etc.)
- Java microservices = heavy dev overhead vs Python
- Research toolkit still in early development
- Enterprise-grade for what may be retail-scale strategies
- 5+ services to manage is overkill for most strategies

## Implementability: 2/5

Powerful but heavy. Only suitable if you need the full reverse-engineering pipeline. Overkill for simple arb.

## Status: QUEUED
