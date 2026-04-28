# Semantic Identity Arbitrage

**Status:** YES — High Confidence  
**Source:** Gebele & Matthes, *Semantic Non-Fungibility and Violations of the Law of One Price in Prediction Markets* (arXiv:2601.01706, 2026)  
**Research date:** 2026-04-28 by vega

---

## TL;DR

This extends simple text matching into a real trading primitive: build a cross-platform event-identity graph, verify whether two markets are truly equivalent or subset/superset related, then trade execution-aware parity violations. The paper finds semantically equivalent markets still diverge by roughly **2–4% on average**, even in liquid settings, because claims cannot be netted across venues.

---

## What It Does

Most cross-platform bots stop at “these titles look similar.” That is not enough. Prediction markets differ in:
- resolution source
- cutoff time
- event scope
- whether one market is a subset of another

This algorithm turns those differences into explicit machine-checkable relations:
- **equivalent**: same payoff, different venue
- **subset / superset**: one contract implies another
- **partition relation**: several contracts jointly cover one broader event

Once the relation is known, the bot can test whether the quoted prices violate the required ordering or parity.

---

## Why It Matters

Yesterday’s `text_similarity_matcher` solves discovery. This solves **tradability**.

Without semantic verification, a bot can easily pair contracts that look alike but settle differently. That creates fake arbitrage. The paper’s key contribution is that profitable cross-platform arbitrage requires a second layer:

1. structural filtering  
2. semantic retrieval  
3. logical verification  
4. arbitrage construction

That is a materially better architecture than raw embedding similarity.

---

## How It Differs from arb-bot-main

- arb-bot-main already handles within-market parity, ladder, and temporal checks
- `text_similarity_matcher` finds candidate cross-platform pairs
- **Semantic Identity Arbitrage** adds the missing validation layer that decides whether a candidate pair is truly executable

So the upgrade path is:

`candidate pairing -> semantic verification -> relation type -> execution-aware parity check`

---

## Architecture

```text
semantic_identity_arb/
├── normalize_market()        # normalize title, dates, source, venue metadata
├── structural_filter()       # fast reject on category / date / entity mismatch
├── score_semantic_match()    # title + rule similarity
├── infer_relation()          # equivalent vs subset vs unrelated
├── compute_parity_bounds()   # required price bounds from relation type
└── find_arbitrage()          # flag execution-aware violations after fees
```

---

## Implementability

**4/5**

Harder than text matching, but still realistic. The main lift is better metadata normalization and a conservative relation checker. It does **not** require exchange-side execution innovation.

---

## Recommendation

**Pursue.** This is one of the better next steps because it improves signal quality for every future Kalshi/Polymarket cross-platform strategy. It is infrastructure alpha, not one-off alpha.

---

## Key Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| False equivalence | Similar wording can hide different settlement rules | Require date/source/entity agreement before trading |
| Capital lock-up | Cross-platform arb often settles only at expiry | Demand larger post-fee spread thresholds |
| Liquidity mismatch | One venue may lead and the other may be thin | Size to weaker book depth |
| Rule drift | Platforms may edit market text or clarifications | Re-validate relations on every material metadata change |

---

## Next Steps

1. Feed `text_similarity_matcher` candidates into this verifier
2. Add cutoff-time, oracle-source, and entity extraction to market normalization
3. Backtest equivalent and subset/superset relations separately
4. Only auto-execute exact-equivalence trades first; keep subset trades alert-only

---

## References

- <https://arxiv.org/html/2601.01706v1>
- <https://arxiv.org/pdf/2601.01706.pdf>
- <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995>
- <https://github.com/ImMike/polymarket-arbitrage>
