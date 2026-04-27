# openclaw-scratchpad

Prediction market trading algorithms, prototypes, and research documentation from vega (Charles trading bot).

> Each algorithm folder contains: how it works, architecture notes, sample code snippets, and links to primary sources.

---

## Algorithms

| Algorithm | Priority | Status |
|---|---|---|
| [Combinatorial Arbitrage](./combinatorial_arb/) | YES — High Confidence | In Progress |
| [Kelly Sizing + Probability Estimation](./kelly_sizing/) | YES — High Confidence | In Progress |
| [Market Making / Spread Capture](./market_making/) | MEDIUM | In Progress |
| [Cross-Market Rebalancing Arbitrage (CMRA)](./cmra_detector/) | YES — High Confidence | Apr 2026 |
| [Text-Similarity Market Matching](./text_similarity_matcher/) | YES — Low Complexity | Apr 2026 |
| [BTC Options IV Probability Estimation](./btc_options_probability/) | YES — Low Complexity | Apr 2026 |

---

## Primary Research Sources

- **Combinatorial Arbitrage Paper:** [arXiv:2508.03474](https://arxiv.org/abs/2508.03474) — M+ extracted via combinatorial arb on Polymarket
- **CMRA + LLM Detection:** [Flashbots discussion](https://collective.flashbots.net/t/arbitrage-in-prediction-markets-strategies-impact-and-open-questions/5198) + [Bawa Medium](https://medium.com/@navnoorbawa/combinatorial-arbitrage-in-prediction-markets-why-62-of-llm-detected-dependencies-fail-to-26f614804e8d)
- **Becker Microstructure:** [jbecier.dev](https://www.jbecker.dev/research/prediction-market-microstructure)
- **BTC Options IV:** [Moontower substack](https://navnoorbawa.substack.com/p/arbitrage-evolution-from-morgan-stanleys)
- **Text-Similarity Matching:** [ImMike/polymarket-arbitrage](https://github.com/ImMike/polymarket-arbitrage)
