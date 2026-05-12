# PIRAP: Resolution-Aware Perpetual Futures Framework

**Source:** https://arxiv.org/abs/2605.10400
**Recommendation:** YES ✅ (cutting-edge research)
**Implementability:** 2/5 (research-grade, not deployment-ready)

## What It Does

Presents a formal framework for **perpetual futures contracts** whose underlying tracks a binary prediction market probability through resolution. Published yesterday (May 11, 2026) — brand new.

Six components:
1. **Index Estimator** — Combines mid-price, depth-weighted mid, and time-decayed VWAP
2. **Jump-Aware Tiered Margin** — Sized against bounded-event terminal-collapse magnitude
3. **Leverage Compression Schedule** — Contracts automatically toward resolution
4. **Resolution-Aware Funding Rule** — Boundary-aware correction prevents manipulation
5. **Multi-Stage Halt Protocol** — Prevents death spirals near resolution
6. **Eligibility Framework** — Which underlyings qualify

**Key finding:** Standard basis-only funding paired with continuous-vol static margin **fails** on bounded-event underlyings. The halt protocol addresses execution-channel risk, but terminal-jump bad-debt remains a margin-side problem.

## Why It Matters

- First formal treatment of **prediction-market perpetuals** — crypto's holy grail for prediction markets
- 86-page paper with full empirical evaluation on PMXT v2 archive (13,298 markets)
- Code available at github.com/ForesightFlow/event-linked-perps
- Identifies fundamental structural limitations of applying standard perpetual frameworks to prediction markets

## Risks

- **Explicitly non-deployable** — the framework does not pass all materiality floors
- Research-grade, not production-ready
- 3 of 5 materiality floors fail
- Requires deep understanding of perpetual mechanics
- No ready-to-run bot — this is research, not code

## Next Steps

- Read the paper in full (86 pages) — especially the empirical evaluation
- Review the companion code on GitHub
- Monitor for v2 of the framework addressing the margin-side bad-debt issue
- This could form the basis for a novel trading product once matured

## Implementability Score: 2/5

Research-grade. Not deployable today, but the structural insights are valuable for anyone building prediction market derivatives.
