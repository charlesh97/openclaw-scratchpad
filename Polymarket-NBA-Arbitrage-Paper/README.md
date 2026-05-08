# Polymarket NBA Arbitrage Paper (arXiv:2605.00864)

**Source:** https://arxiv.org/abs/2605.00864

## What It Is
Empirical academic paper analyzing **75 million Polymarket order book snapshots across 173 NBA games**, reconstructing continuous market states to evaluate the frequency, duration, and profitability of arbitrage opportunities. Published April 2026.

## Key Findings (Critical for Any Arb Strategy)
1. **Single-market anomalies are exceedingly rare** — only 7 executable in-game episodes across all 173 games (median duration: 3.6 seconds)
2. **Combinatorial inefficiencies are more frequent** — 290 active episodes, concentrated in final minutes of live play (abrupt scoring causes cross-market dislocations)
3. **Combinatorial execution yields ~101 bps median return** — statistically meaningful but not "jackpot"
4. **Liquidity is the binding constraint** — 76.9% of combinatorial opportunities had max executable size of only **14.8 shares**; theoretical "jackpot" arb is never empirically realized
5. **Limits-to-arbitrage confirmed** (Shleifer & Vishny 1997 framework): execution frictions prevent full correction of mispricings even when unambiguously identified

## Why NBA Markets Specifically?
- Sports markets resolve within hours (not days/weeks) — less time for arbitrageurs to act
- Scoring events create rapid cross-market dislocations — more arb "surface area"
- But shallow order books mean these dislocations are unexecutable at scale

## Implementability: N/A (Research Paper)
- Read as essential context before spending time on arb infrastructure
- The data confirms what Medium/Twitter threads obscure: simple "buy YES + sell NO" arb is real but structurally limited to small sizes and millisecond windows
- For meaningful profit at scale, cross-platform arb (Polymarket vs Kalshi) has more potential than single-platform bundle arb

## Implications for Strategy
- **Single-platform bundle arb**: Not worth building infrastructure for — too rare, too fast
- **Cross-platform (Poly vs Kalshi)**: More promising — two independent books mean more varied pricing
- **Market making on wide spreads**: The most durable edge; spreads are wide enough to earn before arb compresses them
- **Scale constraint**: Unless you can execute at 14-share size repeatedly, transaction costs dominate

## Next Steps
1. Read the full paper for microstructure detail
2. Use findings to calibrate expectations for any arb strategy
3. Cross-platform arb is the right focus area per the paper's findings
