# Arbitrage Analysis in Polymarket NBA Markets

**Source:** https://arxiv.org/abs/2605.00864
**Published:** April 2026

## Summary

Empirical study of arbitrage in Polymarket's NBA moneyline and spread markets during the 2025-2026 season.

### Key Findings

1. **Simple bundle arbitrage** (YES + NO < $1) is rarest — virtually nonexistent in liquid NBA markets
2. **Combinatorial arbitrage** across Moneyline-Spread pairs is more frequent: **290 active episodes** concentrated in final minutes of live play
3. **Structural bottleneck**: 76.9% of combinatorial opportunities constrained to average executable size of just **14.8 shares** (~$14.80)
4. **Liquidity shallowness** is the primary friction preventing full correction of mispricings

### Limits to Arbitrage
Confirms the Shleifer and Vishny (1997) framework: even when mispricings are identified, execution frictions prevent exploitation at scale. "Risk-free" extraction is confined to retail scale.

## Implication
NBA markets offer genuine but tiny arb opportunities during high-volatility periods (final minutes). Not worth institutional deployment but viable for automated retail bots.

## Next Steps
1. Monitor NBA live games for cross-market dislocations in final minutes
2. Build a lightweight scanner targeting Moneyline-Spread pairs
3. Accept that position sizes will be capped at ~$15 per opportunity
