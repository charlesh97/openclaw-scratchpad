# Summary: Anatomy of a Decentralized Prediction Market

## Dataset
- 30 billion WebSocket order book events
- 52 days of continuous tick-level data
- 600 pre-registered stratified panel markets
- Joined to authoritative on-chain trade records

## 8 Stylized Facts
1. **Longshot spread premium:** Longer odds → wider spreads
2. **Depth concentration:** Uniform geometric grid, not concentrated at top-of-book
3. **Null block-clock effect:** No meaningful difference in microstructure between block times
4. **Maker diversity:** Broad distribution with a concentrated tail (few wallets provide most liquidity)
5. **Category spreads:** Sports vs. crypto vs. politics have different effective spreads
6. **Ingestion delay:** Median <50ms, but tail >2 seconds
7. **Wash trading:** Self-counterparty median 1%, upper tail 22%
8. **Comparable to traditional finance benchmarks**

## Implications for Bots
- Liquidity is distributed across the book — not just top level
- Latency advantage matters but only below 50ms for marginal gains
- Focus on categories with wider spreads (more inefficiency)
