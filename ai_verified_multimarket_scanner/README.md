# AI-Verified Multi-Market Scanner

## What It Does

An arbitrage scanner that monitors **three prediction market platforms simultaneously** (Polymarket, Kalshi, Gemini Predictions), uses **NLP-based text similarity** to match semantically equivalent contracts across platforms, then runs every candidate through an **LLM verification layer** before alerting — eliminating false positives from naive price comparison.

This is a **research-grade scanning framework**, not an execution bot. The core innovation is the AI verification step: instead of blindly trusting that two contracts with similar text are the same market, it asks a language model to confirm the contracts refer to the same outcome before calculating the spread.

## Architecture

```
Platform APIs (Polymarket, Kalshi, Gemini)
       ↓
  Market Fetcher (30s polling interval)
       ↓
  NLP Matcher (TF-IDF / cosine similarity on question text)
       ↓
  Candidate Arb Opportunities
       ↓
  LLM Verifier (Claude API — confirms same outcome)
       ↓
  Fee-Adjusted Profit Calculator
       ↓
  Alert (verified opportunities above threshold)
```

**Core components:**
- `scanner.py` — orchestrates multi-platform polling
- `matcher.py` — NLP-based cross-platform market matching (text similarity)
- `detector.py` — arbitrage condition + fee math
- `claude_verifier.py` — LLM verification before alert
- `position_tracker.rs` (optional Rust module for production)

## How It Differs From arb-bot-main

arb-bot-main uses rule-based market matching (hard-coded team code mappings, slug conventions). This approach uses **semantic text matching** to discover equivalent markets that don't share obvious naming conventions — catching more opportunities but with higher latency from LLM calls.

## Implementability: 4/5

- ✅ Open-source, well-documented Python implementation
- ✅ Modular — easy to plug in new platforms
- ✅ Claude verification layer is genuinely novel
- ⚠️ LLM calls add latency (1–3s per candidate), reduces urgency
- ⚠️ Polling-based (30s), misses fast fleeting opportunities

## Recommendation: MEDIUM-YES

The AI verification layer addresses a real pain point in cross-platform scanning: **false positive arbs from mismatched markets**. This is especially valuable as the ecosystem grows more markets with different naming conventions across platforms. However, the 30s polling and LLM latency make it unsuitable for ultra-short-duration opportunities.

**Best for:** Traders running multiple strategies who want a high-confidence alert system feeding into a broader pipeline.

## Key Sources

- **GitHub:** [blacksyncai/arbitrage-bot](https://github.com/blacksyncai/arbitrage-bot)
- **Concept:** [Polysmart.io](https://www.polysmart.io/) AI oracle
- **Platform fee table:** Polymarket 0.75–1.80%, Kalshi ~2.0%, Gemini ~1.75%

## Risks

1. **LLM latency** — 1–3 second verification delay means opportunities may expire
2. **False negatives** — LLM may incorrectly reject valid opportunities
3. **Cost** — Anthropic API costs accumulate with high candidate volume
4. **Polling gap** — 30s intervals miss sub-30s opportunities

## Next Steps

1. Replace polling with WebSocket feeds for real-time data
2. Add a local embedding model (e.g., sentence-transformers) to replace LLM verification for common markets
3. Build a priority queue that fast-tracks high-confidence matches around LLM calls
4. Integrate with execution layer (e.g., joy-deploy Rust bot) for automated trading