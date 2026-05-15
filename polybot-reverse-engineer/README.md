# Polybot — Strategy Reverse-Engineering Toolkit

**Source:** https://github.com/ent0n29/polybot  
**Recommendation:** YES

## What It Does

A production-grade Polymarket trading infrastructure toolkit implemented in Java 21 microservices. Multi-service system supporting execution (paper + live), strategy runtime, market making, and quantitative analysis. Includes a ClickHouse + Redpanda event pipeline for trade ingestion and replication scoring.

## Architecture

- **Executor Service** — Order placement and execution
- **Strategy Service** — Runtime for trading strategies
- **Ingestor Service** — Market and user trade ingestion into ClickHouse
- **Analytics Service** — Quantitative analysis and replication metrics
- **Infrastructure Orchestrator** — Service coordination

## Key Features

- Full monitoring stack (Grafana, Prometheus, Alertmanager)
- Research toolkit for snapshots, deep analysis, replication metrics
- Foundation for AWARE (trader intelligence, PSI indices, fund mirroring)
- Java 21 microservices with Spring Boot
- Docker Compose for easy deployment

## Why It Matters

This is the most sophisticated open-source Polymarket infrastructure available. The event pipeline architecture and research toolkit make it ideal for quantitative strategy development. The AWARE layer (separate repo) adds retail-facing products.

## Implementability: 2/5

Requires Java 21, Maven, Docker, ClickHouse, Redpanda. Heavy infrastructure. Best suited for a team, not a single developer.

## Risks

- High operational complexity
- Java microservices require DevOps knowledge
- Resource-intensive (requires Docker stack)
