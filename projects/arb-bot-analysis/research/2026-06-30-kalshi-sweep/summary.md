# 2026-06-30 Kalshi Deep Dive — Follow-up Summary

## I. Market Microstructure

### "Makers and Takers" (Whelan et al., Jan 2026)
- **Source:** https://www.karlwhelan.com/Papers/Kalshi.pdf
- 12,403 events, 46,282 contracts, 313,972 prices (2021–Apr 2025)
- Kalshi = decentralized quote-driven market (NOT CLOB)
- Favorite-longshot bias: <$0.10 contracts lose >60%; >$0.50 earn positive returns
- Overall avg return ≈ −20%
- Makers earn more than Takers; both show FLB pattern
- Modest disagreement + probability overstatement → reproduces bias

### "Domain-Specific Calibration" (Le, arXiv:2602.19520)
- **Source:** https://arxiv.org/abs/2602.19520
- 292M trades, 327K contracts, Kalshi + Polymarket
- 4 components explain 87.3% variance:
  - Universal horizon effect: 30.2%
  - Domain-specific biases: ~35%
  - Domain × horizon interactions: ~17%
  - Trade-size scale effect: ~5%

**Calibration slopes (Kalshi, by domain and trade size):**

| Domain | Single | Small | Medium | Large | Δ(L−S) |
|--------|--------|-------|--------|-------|--------|
| Politics | 1.19 | 1.22 | 1.37 | 1.74 | +0.53 |
| Sports | 1.00 | 1.01 | 1.01 | 1.01 | +0.07 |
| Crypto | 1.03 | 1.03 | 1.02 | 1.00 | −0.02 |
| Finance | 1.10 | 1.08 | 1.05 | 1.05 | −0.05 |
| Weather | 0.96 | 0.94 | 0.91 | 0.89 | −0.07 |
| Entertainment | 0.98 | 1.02 | 1.00 | 0.99 | +0.01 |

Slope >1 = underconfidence. Key: politics Large trades +0.53 on Kalshi, NOT on Polymarket (+0.11).

3 stylized facts:
1. All markets underconfident at long horizons (FBL bias)
2. Different domains = different calibration trajectories
3. Large trades amplify underconfidence in politics on Kalshi only

Concrete: 70¢ political contract 1wk before resolution ≈ 83% true probability.

## II. Market Inefficiency

### "When Do Markets Process Public Information?" (Angelini & De Angelis, arXiv:2606.07811)
- **Source:** https://arxiv.org/abs/2606.07811
- NBA live contracts on Kalshi
- Prices adjust 0.64-for-one vs benchmark → missing 0.36 predicts drift
- Salience × liquidity interaction drives underreaction
- Thin markets + salient signals = largest drift

### NegRisk Rebalancing ($29M)
- **Source:** https://navnoorbawa.substack.com/p/negrisk-market-rebalancing-how-29m
- Buy NO = 60% of profits
- 29× capital efficiency vs binary arb
- Multi-condition: Σ(prices) must = 1.0

## III. Infrastructure

### Kalshi API
- **Source:** https://docs.kalshi.com/
- REST + WebSocket + FIX
- Perps API separate
- Demo environment available
- RSA-PSS auth

### Fees
- **Source:** https://kalshifees.com/
- 1-7¢ per contract, highest at 50/50
- WAIVED on losing trades
- Makers charged since Apr 2025

### GitHub Repos

| Repo | Stars | Language | Status | Best For |
|------|-------|----------|--------|----------|
| TopTrenDev/polymarket-kalshi-arbitrage-bot | 42 | Rust | Active (Jun 2026) | Cross-platform Rust |
| antmlap/kalshi-arbitrage-bot | 7 | Python | Active (May 2026) | Kalshi-only Python |
| rao/pm-kalshi-arbitrage-bot | 0 | TypeScript | Inactive | Skip |

## IV. Recommended Starting Points

1. Read Whelan paper first (Kalshi operating manual)
2. Run antmlap bot in dry-run (clean Python, Kalshi-only)
3. Focus on political markets (largest calibration bias)
4. For sports: 0.64 underreaction + drift is clean alpha signal
5. For cross-platform: TopTrenDev Rust bot as reference architecture
6. Model Kalshi fees explicitly (different from Polymarket)
7. NegRisk rebalancing is underexploited on Kalshi specifically
