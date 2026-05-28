# Summary: Arbitrage Analysis in Polymarket NBA Markets

**arXiv:2605.00864 | April 2026 | Cheng, Yang et al.**

This paper empirically examines whether algorithmic arbitrage is actually profitable on Polymarket. Using 75M order book snapshots from 173 NBA games, the authors find:

- Single-market arbitrage: 7 episodes, ~3.6s median duration — essentially non-existent
- Combinatorial arbitrage: 290 episodes, tied to final minutes of games
- Median return: 101 bps, but max size ~15 shares due to shallow books
- **Key insight:** The market is microstructurally efficient for retail-sized positions; arb is bounded by liquidity, not detection

**Bottom line:** Useful for understanding the structural limits of arb, not as a deployable trading strategy.
