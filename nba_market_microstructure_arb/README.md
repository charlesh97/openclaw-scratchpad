# NBA Market Microstructure Arbitrage
## Polymarket NBA LOB Arbitrage — Order Book Deep-Dive Strategy

**arXiv:** [2605.00864](https://arxiv.org/abs/2605.00864) — "Arbitrage Analysis in Polymarket NBA Markets"
Guang Cheng, Jiaxin Yang, Haoxuan Zou (UCLA), May 2026

---

## What It Does

Analyzes **75M+ limit order book (LOB) snapshots** across 173 NBA games to systematically quantify real arbitrage opportunities in Polymarket's CLOB. Classifies arb types:
- **Single-market arb**: YES + NO within one market sum < $1.00
- **Combinatorial arb**: Mispricings across multiple related markets (e.g., game winner + first quarter winner)

## Key Empirical Findings

| Metric | Value |
|---|---|
| LOB snapshots analyzed | 75,088,497 |
| Games covered | 173 NBA games |
| Single-market arb episodes | 7 (rare) |
| Single-market median duration | **3.6 seconds** |
| Combinatorial arb episodes | 290 |
| Combinatorial median return | **101 bps** |
| Avg executable size (combinatorial) | **14.8 shares** (76.9% limited to retail scale) |
| "Middle jackpot" (theoretical max) | **Never empirically realized** |

> **Core insight:** Executable mispricings exist but are **structurally bounded by liquidity**. The theoretical arbitrage is real; execution at scale is not viable for institutional capital.

---

## Architecture

1. **LOB Snapshot Collector** — Reconstruct continuous market state from Polymarket's CLOB at high resolution
2. **Single-Market Arb Detector** — Scans each market for YES + NO sum < $1.00
3. **Combinatorial Pair Scanner** — Finds mispricings across logically related markets (same game, different props)
4. **Liquidity Filter** — Flags only opportunities with sufficient depth to execute
5. **Execution Estimator** — Reports achievable position size vs. theoretical size

## Implementability: 2/5
- Requires **high-frequency LOB access** (not available via public API; need WebSocket or CLOB API with market-making permissions)
- Built with proprietary dataset (75M snapshots) — not reproducible without the data
- Retail-scale execution only; institutional capital hits liquidity wall immediately
- **Recommendation:** Use as a **liquidity awareness layer** — don't trade into shallow combinatorial markets assuming large size; limit position sizing to what LOB depth supports

## Risks
- **Liquidity risk**: 76.9% of combinatorial opportunities are constrained to <15 shares
- **Execution speed**: Single-market opportunities persist only 3.6 seconds — requires co-location or sub-second infrastructure
- **Scale trap**: The strategy confirms arb exists but proves institutional execution is not feasible at theoretical returns

## Next Steps
- Add LOB depth monitoring to existing arb-bot: only flag opportunities where both legs have ≥ X shares at the arb price
- Build a "liquidity-adjusted edge" calculator: (theoretical edge) × (max_executable_shares) to find actual achievable profit per opportunity
- Use findings to filter combinatorial arb candidates — avoid markets where depth is too shallow

## Source
- arXiv: https://arxiv.org/abs/2605.00864
- HTML: https://arxiv.org/html/2605.00864
