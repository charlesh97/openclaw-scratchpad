# Semantic Non-Fungibility Framework
## Cross-Platform Event Identity Resolution + Persistent Arbitrage

**arXiv:** [2601.01706](https://arxiv.org/abs/2601.01706) — "Semantic Non-Fungibility and Violations of the Law of One Price in Prediction Markets"
Gebele, Matthes — Technical University of Munich, January 2026

---

## What It Does

Identifies a **fundamental structural barrier**: prediction markets define the same underlying event through platform-specific natural language descriptions. Without a shared notion of event identity:
- Liquidity fails to pool across venues
- Arbitrage requires capital commitment until resolution (not truly "risk-free")
- Prices systematically violate the Law of One Price

The paper introduces a **semantic alignment framework** using NLP to match equivalent events across platforms — and then quantifies how much money is left on the table due to this friction.

## Key Empirical Findings

| Metric | Value |
|---|---|
| Events analyzed | 100,000+ across 10 venues (2018–2025) |
| Events listed on multiple platforms | ~6% |
| Average execution-aware price deviation | **2–4%** |
| Driver of persistent arb | **Structural frictions**, not informational disagreement |

> **Core insight:** 2–4% persistent cross-platform mispricings are driven by **semantic non-fungibility** — not irrational markets. Fix the event identity problem and these arb opportunities become capital-efficient. Until then, they require capital commitment and carry resolution risk.

---

## Architecture

1. **Event Description Parser** — Extracts natural-language event description from each market
2. **Semantic Embedder** — Joint analysis of: description text, resolution semantics, temporal scope
3. **Cross-Platform Matcher** — Aligns markets across Kalshi, Polymarket, PredictIt, etc.
4. **Price Deviation Monitor** — Tracks aligned pairs, flags when deviation > transaction cost threshold
5. **Capital-at-Risk Calculator** — Computes profit adjusted for capital lock-up duration

## Implementability: 3/5
- Semantic alignment framework requires NLP pipeline (embeddings + similarity thresholding)
- Cross-platform market discovery across 10 venues is data-intensive
- 2–4% average deviation is large enough to be attractive even after fees
- **Recommendation:** Build as a **pre-trade filter** — before attempting cross-platform arb, check if the event pair is semantically confirmed equivalent. False equivalences are the #1 risk in cross-platform arb.

## Risks
- **Semantic mismatch risk**: Two markets that LOOK equivalent may resolve differently (e.g., "Which party controls the Senate?" vs "Republican Senate majority" — different resolution windows)
- **Capital lock-up risk**: Unlike single-market arb (instant settlement), cross-platform requires holding positions until resolution — carries overnight/event risk
- **Oracle risk**: Same event resolved by different oracles can produce different outcomes
- **Regulatory risk**: Cross-platform position netting may have regulatory implications

## Next Steps
- Build a semantic similarity scorer using sentence embeddings (e.g., sentence-transformers) to flag cross-platform event equivalence before attempting arb
- Add resolution rules comparison: not just "same event" but "same oracle + same cutoff time"
- Create a cross-platform event graph database to track which venues list the same events

## Source
- arXiv: https://arxiv.org/abs/2601.01706
- HTML: https://arxiv.org/html/2601.01706v1
