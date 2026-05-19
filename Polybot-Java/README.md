# Polybot — Java Microservices Trading Infrastructure

**Source:** [ent0n29/polybot](https://github.com/ent0n29/polybot)
**Recommendation:** NO (queue)

## What It Does

An open-source Polymarket trading infrastructure and strategy reverse-engineering toolkit built with Java 21 microservices. Includes:

- **Executor Service** — Automated execution (paper + live)
- **Strategy Service** — Strategy runtime and market making
- **Ingestor Service** — Market/user trade ingestion into ClickHouse
- **Analytics Service** — Quantitative analysis and replication scoring

### Architecture
```
Java 21 Microservices
├── Executor (trade execution)
├── Strategy (runtime + MM)
├── Ingestor (ClickHouse pipeline)
└── Analytics (replication scoring)

Infrastructure: ClickHouse + Redpanda + Grafana + Prometheus
```

## Implementability: 2/5

- **Java 21** — significant departure from Python/Node.js stack
- Complex infrastructure (ClickHouse, Redpanda, microservices)
- Too heavy for our current needs

## Risks
- Operational complexity is disproportionate to expected gains
- Java talent/preference may differ from our stack

## Next Steps
Keep on watchlist. The strategy reverse-engineering concept is interesting but better implemented in Python.
