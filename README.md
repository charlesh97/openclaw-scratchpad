# openclaw-scratchpad

Prediction market trading algorithms, prototypes, and research documentation from vega (Charles' trading bot).

> This repo is the implementation counterpart to the daily morning research reports sent via email. Each algorithm folder contains: how it works, architecture notes, sample code snippets, and links to primary sources.

---

## Algorithms

| Algorithm | Priority | Status |
|---|---|---|
| [Combinatorial Arbitrage](./combinatorial_arb/) | ✅ YES — High Confidence | In Progress |
| [Kelly Sizing + Probability Estimation](./kelly_sizing/) | ✅ YES — High Confidence | In Progress |
| [Market Making / Spread Capture](./market_making/) | MEDIUM | In Progress |

---

## Primary Research Sources

- **Combinatorial Arbitrage Paper:** [arXiv:2508.03474 — "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"](https://arxiv.org/abs/2508.03474)
  - Raw PDF: <https://arxiv.org/pdf/2508.03474>
  - Authors: Saguillo, Ghafouri, Kiffer, Suarez-Tangil
  - Key finding: $40M+ of profit extracted via combinatorial arbitrage on Polymarket
- **Becker Microstructure Paper:** <https://www.jbecker.dev/research/prediction-market-microstructure>
- **Price Discovery Paper (Ng et al.):** <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995>

---

## Repo Structure

```
openclaw-scratchpad/
├── README.md
├── combinatorial_arb/
│   ├── README.md
│   ├── architecture.md
│   └── sample_code.py
├── kelly_sizing/
│   ├── README.md
│   ├── architecture.md
│   └── sample_code.py
└── market_making/
    ├── README.md
    ├── architecture.md
    └── sample_code.py
```

---

## Architecture Overview

All algorithms are designed to slot into the existing `arb-bot-main` codebase as detached modules:

```
arb-bot-main/
├── algos/
│   ├── common/
│   │   ├── base_bot.py
│   │   └── opportunity.py
│   ├── parity/         ← existing
│   ├── temporal/       ← existing
│   └── combinatorial/  ← NEW: this repo
```

Each module follows the same interface as existing `check_*` functions:
1. Input: market data (price, volume, end_time, conditions)
2. Output: list of `Opportunity` objects with score, thesis, risk, invalidation
3. Guard rails: position limits, kill switch, regime filters

---

## Status

Updated daily by vega as part of the morning research workflow.
