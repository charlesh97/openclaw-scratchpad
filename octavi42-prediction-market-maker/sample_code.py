"""
octavi42/prediction-market-maker — LMSR Market Making Concept
Based on: https://github.com/octavi42/prediction-market-maker

Reference implementation of LMSR (Logarithmic Market Scoring Rule) market maker.
Paradigm Prediction Market Challenge #2 entry.
"""
import math

def lmsr_cost(b, q, B):
    """
    Calculate cost to move the market probability.
    b: current best prices (vector)
    q: current shares outstanding (vector)
    B: market maker budget (controls spread)
    """
    return B * math.log(sum(math.exp(q[i] / B) for i in range(len(q))))

def lmsr_price(q, B, outcome_idx):
    """
    Compute implied probability price for a given outcome.
    q: shares vector
    B: market maker budget
    outcome_idx: which outcome to price
    """
    return math.exp(q[outcome_idx] / B) / sum(math.exp(q[i] / B) for i in range(len(q)))

def optimal_spread(market_volatility, adverse_selection_prob, B=1000):
    """
    Heuristic for setting bid-ask spread:
    spread >= cost of adverse selection per trade
    """
    return 2 * adverse_selection_prob * market_volatility

# Key insight: market making earns the spread, not the direction
# Spread must be wide enough to cover adverse selection cost from informed traders
