# Semantic Market Clustering

**Source:** arXiv:2512.02436 — "Semantic Trading: Agentic AI for Clustering and Relationship Discovery in Prediction Markets" (Dec 2025)

## What It Does

An end-to-end LLM-based pipeline that reads market text, clusters markets into groups with correlated or interdependent outcomes, and identifies combinatorial relationships that create arbitrage opportunities. Unlike simple text similarity, this approach discovers logical dependencies (e.g., "If X wins presidency, then X's party likely wins Senate").

## How It Differs from Our Current Stack

| Aspect | Text Similarity Matcher | Semantic Market Cluster |
|--------|------------------------|------------------------|
| Method | Cosine similarity on embeddings | LLM clustering + relationship extraction |
| Output | Pairs of similar markets | Clusters with dependency graphs |
| Dependency types | Surface-level text match | Logical, causal, temporal dependencies |
| Scalability | Pairwise O(n²) | Cluster-based, more efficient |
| False positive rate | High (semantic similarity ≠ dependence) | Lower (LLM validates logical links) |

## Architecture

```
1. Ingestion
   All Polymarket/Kalshi markets ──→ Market Text Corpus

2. Clustering Phase
   Market Text Corpus ──→ LLM Embedding ──→ Semantic Clustering
                                              (group by topic/domain)

3. Relationship Extraction
   Each Cluster ──→ LLM Dependency Analysis ──→ Dependency Graph
                  (identify which markets          (A implies B,
                   logically affect each other)     B contradicts C,
                                                    A and B are complements)

4. Arbitrage Detection
   Dependency Graph ──→ Constraint Violation Check ──→ Arbitrage Signals
                      (do probabilities obey the
                       logical constraints?)
```

## Key Findings from Paper

- LLM agents can discover non-obvious inter-market dependencies that text similarity misses
- Clustering reduces the search space from O(n²) to O(clusters × cluster_size)
- Dependencies are categorized as: implications, contradictions, complements, conditionals
- Cross-platform analysis (Polymarket vs Kalshi) reveals structural mispricings
- Pipeline processes 1000+ markets in minutes

## Implementability: 3/5

**Pros:**
- More accurate dependency detection than our current text similarity approach
- Reduces combinatorial explosion in cross-market analysis
- Can discover novel dependency types (conditional, temporal)
- Integrates well with existing CMRA and NegRisk detectors

**Cons:**
- LLM dependency extraction can hallucinate false relationships
- Requires periodic re-clustering as new markets appear
- Clustering quality depends on embedding model choice
- More complex pipeline than current approach

## Risks

1. **Hallucinated dependencies:** LLM might claim markets are related when they're not
2. **Stale clusters:** Market semantics evolve — clusters need refreshing
3. **Computational cost:** LLM processing of 1000+ markets is not free
4. **Over-smoothing:** Clustering might merge genuinely distinct markets

## Next Steps

1. **Quick win:** Use our existing embeddings + LLM to extract dependency graphs from top 100 markets
2. **Compare:** Run text similarity matcher vs semantic clustering on same dataset
3. **Integrate:** Feed dependency graphs into CMRA detector for better combinatorial arb detection
4. **Benchmark:** Measure false positive rate of dependency extraction

## Reference

- Paper: https://arxiv.org/abs/2512.02436
- Authors: Not specified in search results (Dec 2025)
- Key technique: LLM-based market clustering + dependency graph extraction
