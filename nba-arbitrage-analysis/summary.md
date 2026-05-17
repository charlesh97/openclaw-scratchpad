# Summary: Arbitrage Analysis in Polymarket NBA Markets

## Dataset
- 75 million+ limit order book snapshots
- 173 NBA games
- Both Moneyline and Spread markets

## Key Results
| Metric | Single-Market | Combinatorial |
|--------|--------------|--------------|
| Executable episodes | 7 | 290 |
| Median duration | 3.6 seconds | Final minutes |
| Median return | N/A | 101 bps |
| Avg executable size | N/A | 14.8 shares |
| Liquidity-constrained | N/A | 76.9% |

## Key Insight
"While executable mispricings exist, they are structurally bounded by liquidity, confining risk-free extraction strictly to the retail scale."

## Implications
- Sports arb is possible but small-scale
- Combinatorial arb (Moneyline × Spread) is the better opportunity
- Focus on final minutes of live play when dislocations spike
- Automated small-size execution (14.8 shares avg) is the right approach
