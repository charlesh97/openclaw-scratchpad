# Summary: ForesightFlow — Information Leakage Score Framework

**arXiv:2605.00493** — Nechepurenko, May 2026

## TL;DR

A mathematical framework to detect insider trading on prediction markets. ILS score measures how much of the eventual price move happened before the news. Three key findings about proxy quality, timestamp anchoring, and why documented Polymarket insider cases fall outside the original ILS scope.

## Key Concepts

| Concept | Description |
|---------|-------------|
| ILS Score | Fraction of terminal move priced in before public event |
| Scope Conditions | Edge effect, non-trivial move, anchor sensitivity |
| Deadline-ILS | Extension for deadline-resolved markets |
| Resolution Typology | Classification of 911,237-market corpus |

## Practical Takeaways

- Don't trust resolution-anchored timestamps for ILS calculations
- Proxy quality is the binding constraint for insider detection
- Most documented Polymarket insider cases are deadline-resolved
- Deadline-ILS extension closes the gap between methodology and reality
