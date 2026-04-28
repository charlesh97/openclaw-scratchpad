# NegRisk Multi-Condition Rebalancing Arbitrage

## What It Is

Kalshi's **NegRisk** markets (also called "multi-condition" or "conditional" markets) structure a single question as a set of mutually exclusive, exhaustive conditions. For example:

> "Will the Fed funds rate be held, cut 25bp, or cut 50bp+ by Dec 2025?"
> → Condition A: Hold, Condition B: Cut 25bp, Condition C: Cut 50bp+

Each condition resolves to exactly one outcome, so their probabilities must sum to $1.00. The arbitrage opportunity: when the market-implied probabilities of the YES contracts for each condition don't sum to $1.00 correctly — or when logical relationships between conditions are violated.

**Key insight from IMDEA paper (arXiv 2508.03474):** Within-market arbitrage ($40M extracted) consists of two subtypes:
- **Type 1 (already covered):** YES + NO on the same single-condition contract sums to < $1.00
- **Type 2 (this strategy):** Multi-condition markets where the YES contracts for complementary conditions are mispriced relative to each other, or where negation-pair relationships are violated

## How It Differs From arb-bot-main

arb-bot-main's `check_range()` handles multi-outcome markets by verifying that bid-ask spreads don't cross. The **NegRisk MRA** extends this with:
1. Cross-condition logical constraint checking (e.g., if Condition A implies Condition B, then P(A) ≤ P(B))
2. Intra-condition parity checking (the sum of all condition probabilities should = $1.00)
3. **Negation pair detection:** In Kalshi's NegRisk markets, certain conditions are logical negations of each other. Price violations here are direct arb.

## Implementation

### Core Logic

For a NegRisk market with N conditions `{C1, C2, ..., CN}`:

1. **Sum check:** Σ P(Ci) should = 1.00. Flag when |Σ P(Ci) - 1.00| > fee_threshold
2. **Monotonicity:** If Ci ⊂ Cj (Ci logically implies Cj), then P(Ci) ≤ P(Cj)
3. **Negation pairs:** If Ci = ¬Cj, then P(Ci) + P(Cj) = 1.00

### Kalshi NegRisk Market Structure

Kalshi's API exposes multi-condition markets via the `conditions` array in market objects. Each condition has:
- `condition_id`: unique identifier
- `question`: the condition question text
- `outcome`: the specific outcome value

### Feasibility: 3/5

Requires Kalshi API access and ability to parse multi-condition market structures. The heuristic is straightforward but Kalshi's API doesn't always expose condition relationships explicitly — may require NLP inference from question text.

## Risks

- **Kalshi fee structure** can eliminate the edge on small mispricings
- **Execution latency**: Kalshi NegRisk markets may have wider spreads and lower liquidity
- **Condition parsing**: inferring logical relationships from question text is error-prone

## Next Steps

1. Parse Kalshi `/markets` API response for multi-condition market structures
2. Implement sum-check and monotonicity validator
3. Backtest on historical Kalshi NegRisk data (Becker dataset: jon-becker/prediction-market-analysis)
