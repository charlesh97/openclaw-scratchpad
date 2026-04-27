# Text-Similarity Cross-Platform Market Matching

**Status:** YES — High Confidence, Low Complexity  
**Source:** ImMike/polymarket-arbitrage (GitHub, 2025-2026)  
**Research date:** 2026-04-27 by vega

---

## TL;DR

Automatically find matching prediction markets across Polymarket and Kalshi using text embeddings + cosine similarity — no manual URL passing required. Scans 10,000+ markets, pairs likely matches, feeds them into arbitrage detection. Replaces manual market pairing with a scalable algorithm.

---

## The Problem It Solves

arb-bot-main matches markets explicitly (by ID or URL). This doesn't scale to the full universe — you can't manually enumerate 10,000+ market pairs. ImMike's bot solves this by embedding every market title + description and computing semantic similarity:

1. Fetch all market titles/descriptions from Polymarket + Kalshi APIs
2. Encode each with a sentence-transformer model (e.g., `all-MiniLM-L6-v2`)
3. For each Polymarket market, compute cosine similarity to all Kalshi markets
4. If top match similarity > 0.85 → flag as a candidate pair
5. Feed candidate pairs into CMRA or standard parity checks

---

## How It Works

```
text_similarity_matcher.py
├── MarketEmbedding  — encodes market title + description → vector
├── PairMatcher      — computes cosine similarity across all market pairs
└── get_candidate_pairs() — returns list of (Polymarket_market, Kalshi_market, similarity_score)
```

The key threshold parameter is `SIMILARITY_THRESHOLD`:
- ≥ 0.90: high confidence match
- 0.85–0.90: plausible match → manual review before execution
- < 0.85: too noisy → ignore

---

## Sample Code (core logic)

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_matching_pairs(
    polymarket_markets: list[dict],
    kalshi_markets: list[dict],
    model,  # sentence-transformer model
    threshold: float = 0.85
) -> list[tuple[dict, dict, float]]:
    """
    Find cross-platform market matches using cosine similarity on embeddings.
    Returns list of (poly_market, kalshi_market, similarity_score) where score >= threshold.
    """
    # Build corpus: market titles + descriptions
    poly_texts = [f"{m['title']} {m.get('description', '')}" for m in polymarket_markets]
    kalshi_texts = [f"{m['title']} {m.get('description', '')}" for m in kalshi_markets]

    # Encode all texts
    poly_embeddings = model.encode(poly_texts, convert_to_numpy=True)
    kalshi_embeddings = model.encode(kalshi_texts, convert_to_numpy=True)

    # Compute similarity matrix (N_poly x N_kalshi)
    sim_matrix = cosine_similarity(poly_embeddings, kalshi_embeddings)

    # Extract above-threshold pairs
    candidates = []
    for i, poly_m in enumerate(polymarket_markets):
        for j, kalshi_m in enumerate(kalshi_markets):
            score = sim_matrix[i, j]
            if score >= threshold:
                candidates.append((poly_m, kalshi_m, score))

    # Sort by similarity descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates
```

---

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.85 | 0.90+ for confidence; 0.85 for broader scan |
| `model_name` | `all-MiniLM-L6-v2` | Fast, good quality; swap for larger model if needed |
| `max_pairs_per_platform` | 500 | Limit memory for large universes |

---

## Risks and Caveats

| Risk | Mitigation |
|---|---|
| False positives (semantically similar ≠ same event) | Manual review tier at 0.85–0.90; auto-execute only ≥ 0.90 |
| Model bias for certain topics | Test on historical market pairs; retune threshold per category |
| Same outcome worded differently | Use description field, not just title, for richer context |

---

## Next Steps

1. Pull Polymarket + Kalshi market lists via their public APIs
2. Encode with `sentence-transformers` (pip install)
3. Run against historical data to calibrate threshold
4. Feed high-confidence pairs into CMRA detector

---

## References

- [ImMike/polymarket-arbitrage](https://github.com/ImMike/polymarket-arbitrage)
