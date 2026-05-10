# Paradigm Prediction Market Maker (#2 Place)

**Source:** https://github.com/octavi42/prediction-market-maker
**Type:** Market-making strategy
**Recommendation:** MEDIUM

## What It Does

Market-making strategy that placed **#2 in Paradigm's Prediction Market Challenge** — 110 iterations in 8 hours. The challenge involved competing to build the best market-making strategy for prediction markets.

The paper covers:
- **Quoting mechanics** — How to set bid/ask prices around the true probability
- **Adverse selection** — What happens when informed traders systematically trade against you
- **Inventory risk** — Managing your exposure when prices move against your position
- **Order sizing** — How much to quote at each price level

## Why It Matters

Understanding market-making theory is essential for any prediction market trading strategy, even if you primarily do arb rather than LP. The core insight: market makers profit by capturing the spread between uninformed flow (random noise) and true probability (signal). When traders with better information (informed flow) trade against you, you lose.

## Key Structural Insights

1. **Spread capture** — Place bids below true probability and asks above; earn the spread when noise traders cross
2. **Adverse selection mitigation** — Don't size up aggressively when you see large order flow (informed traders know something)
3. **Inventory management** — If you're long too many YES shares, adjust quotes to offload inventory before the event resolves
4. **Position limits** — Set hard limits on inventory to prevent single-event catastrophe

## Implementability: 3/5

Good theoretical foundation and reference code, but no production deployment pipeline described. Best used as a reference architecture for building your own market-making logic.

**Next steps:**
1. Read the code and understand the quoting logic
2. Implement the inventory tracking module
3. Add position limit checks
4. Backtest against historical Polymarket data