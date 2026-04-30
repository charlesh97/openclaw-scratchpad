# LLM Dependency Graph for Combinatorial Arbitrage

## What It Does

Builds a semantic dependency graph of prediction markets using LLM analysis of market question text. The graph identifies logically related markets (implications, complements, contradictions) and surfaces only the high-probability dependency pairs for targeted combinatorial arbitrage scanning.

## The Problem It Solves

The IMDEA paper (arXiv:2508.03474) identifies a core scalability challenge:

> "A naive analysis requiring O(2^(n+m)) comparisons" across n conditions in market A and m in market B.

With 10,000+ active markets, brute-force all-pairs comparison is computationally intractable and produces massive false positive rates.

## How It Works

```
Market Text Corpus (10,000+ questions)
        ↓
  Text Embedding Model (e.g., sentence-transformers)
        ↓
  Semantic Similarity Pre-filter  ← reduces from O(n²) to O(n × k)
        ↓
  LLM Dependency Classifier
    "Does Market A's resolution logically imply Market B's outcome?"
    "Are these markets complements or contradictions?"
        ↓
  Dependency Graph (Directed acyclic graph of market relationships)
        ↓
  Arbitrage Scanner (targeted CMRA only on connected nodes)
        ↓
  Actionable Opportunities
```

## Key Findings From Research

- IMDEA uses LLM (Linq-Embed-Mistral) for semantic market matching
- Their heuristic reduction strategy: **timeliness + topical similarity + combinatorial relationships**
- 62% of LLM-detected dependencies fail to generate profit (Medium article, Nov 2025)
- Successful dependency pairs yield $95K+ in documented combinatorial arb (same period)
- Key discriminator: topical/issue similarity + temporal overlap > text-only similarity

## Architecture

The graph is built in three layers:

**Layer 1 — Similarity Pre-filter**
- Encode all market question texts with `sentence-transformers/all-MiniLM-L6-v2`
- Compute cosine similarity; only pairs above threshold (e.g., 0.78) advance
- This cuts the comparison space by ~99%

**Layer 2 — LLM Dependency Classifier**
- Prompt the LLM with the market question pair and a structured schema:
  ```
  Given: Market A = "Will BTC hit $100k by Dec 2025?"
        Market B = "Will BTC close above $95k by Nov 2025?"
  Is there a dependency?
  Type: [IMPLIES | COMPLEMENT | CONTRADICTION | NONE]
  Confidence: [HIGH | MEDIUM | LOW]
  Reasoning: <2 sentence explanation>
  ```
- This is the semantic risk manager — it prioritizes relationships that generalize

**Layer 3 — Arbitrage Opportunity Scanner**
- Given a confirmed dependency edge (A → B), check if conditional probability holds:
  - If A=YES implies B=YES: then P(B) >= P(A)
  - Violation: P(B) < P(A) → combinatorial arb exists
- Size the position based on the probability gap

## Implementation Difficulty

**4/5** — High. Requires:
- Embedding model infrastructure (GPU helpful but not mandatory)
- LLM API access (or local model like Llama 3)
- Graph database or in-memory graph for relationship storage
- Integration with CMRA scanner for the final opportunity check

## Differences From Existing Algorithms

| Algorithm | Approach | This Algorithm |
|---|---|---|
| Semantic Market Cluster | Agentic pipeline for dependency discovery | Graph-based explicit edge classification |
| Semantic Identity Arb | Exact/partial text match for equivalent events | **Semantic dependency detection** (implied, not identical) |
| CMRA Detector | Scans all pairs within a dependency | **Discovers which pairs have dependency in the first place** |

This is a **pre-processing stage** for the CMRA — it makes the CMRA scanner targeting by reducing the search space dramatically.

## Risks

1. **False positive edges:** LLM may flag spurious relationships → wasted CMRA scans
2. **False negative edges:** Missing real dependencies → missed arb
3. **LLM cost:** Per-pair classification is expensive at scale → batch with similarity pre-filter
4. **Stale edges:** Market relationships change as events evolve → graph TTL/expiry needed
5. **P(high) bias:** LLM tends toward HIGH confidence even when wrong → calibration needed

## Performance Characteristics

- Pre-filter: reduces O(n²) to O(n × k) where k ~ 20–50 candidate neighbors per market
- LLM classification: ~0.5s per pair (async batch), can parallelize to ~1000 pairs/min
- Graph traversal for CMRA: O(edges) not O(nodes²)
- End-to-end latency: ~15 min for full 10,000 market corpus (with batching)

## Next Steps

1. **Build market corpus pipeline** — fetch + dedupe + encode market texts
2. **Run similarity pre-filter** — calibrate threshold against known dependency pairs
3. **Batch LLM classification** — build labeled training set from IMDEA dependency pairs
4. **Graph construction** — store edges in SQLite or NetworkX
5. **CMRA integration** — wire graph edges into the existing CMRA scanner
6. **Backtest** — measure precision/recall of edges vs. actual profitable arb opportunities

## References

- [arXiv:2508.03474](https://arxiv.org/abs/2508.03474) — Saguillo et al. (IMDEA, 2025) — heuristic reduction strategy
- [arXiv:2602.07048](https://arxiv.org/abs/2602.07048) — "LLM as Risk Manager" (Feb 2026) — LLM semantic filtering
- [arXiv:2512.02436](https://arxiv.org/abs/2512.02436) — "Semantic Trading" (Dec 2025) — market clustering pipeline
- [Medium: Combinatorial Arb Fail Rate](https://medium.com/@navnoorbawa/combinatorial-arbitrage-in-prediction-markets-why-62-of-llm-detected-dependencies-fail-to-26f614804e8d) — Nov 2025
