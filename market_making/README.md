# Market Making / Spread Capture

**Status:** MEDIUM — High confidence but requires operational infrastructure  
**Sources:** Becker "Microstructure of Wealth Transfer" (2026), ImMike/polymarket-arbitrage  
**Research date:** 2026-04-26 by vega

---

## What It Is

Place competitive bid/ask orders on Kalshi (acting as the market maker/seller) to capture the spread — particularly in high-bias categories where takers systematically overpay for YES contracts.

Becker (2026) documents that takers on Kalshi lose ~57% EV on longshot YES contracts (contracts priced 1–15¢). This happens because:
1. Takers disproportionately buy YES (the "optimism tax")
2. Makers selling YES near longshot prices collect a structural premium
3. The spread between what takers pay and what makers receive is the edge

**The maker-side trade:**
- Sell YES at 5¢ (short the contract)
- If YES wins: you pay $1 → lose $0.95 net
- If YES loses: the NO holder receives $1, and the NO you bought at 95¢ pays out → you receive ~5¢ net
- Over many longshot trades: the ~5¢ average profit per contract compounds

**Sports = 72% of Kalshi volume** with the highest taker bias — making it the best target category.

---

## Architecture

```
arb-bot-main/
├── market_making/
│   ├── order_manager.py     ← places + manages bid/ask orders
│   ├── inventory.py         ← tracks net position per market
│   ├── spread_calculator.py ← computes competitive spread
│   └── risk_gate.py          ← kill switch + loss limits
```

**Integration:** This is a separate execution mode — not a scan-and-alert like `check_parity()`. It requires:
1. Real-time order book data
2. Inventory tracking per contract
3. Position limits per market + per day
4. Execution layer with cancel/replace capability

---

## Sample Code

```python
# ── Spread capture logic ────────────────────────────────────────────

def compute_sell_price(market_mid: float, spread_bps: int = 50) -> float:
    """
    Compute competitive ask price for selling YES.
    market_mid: current mid price (e.g. 0.07 for 7¢ YES)
    spread_bps: spread in basis points (50bps = 0.5%)
    Returns the ask price (what you sell at).
    """
    return market_mid * (1 - spread_bps / 10_000)


def compute_buy_price(market_mid: float, spread_bps: int = 50) -> float:
    """
    Compute competitive bid price for buying YES (to hedge inventory).
    """
    return market_mid * (1 + spread_bps / 10_000)


def maker_pnl_per_contract(yes_price_sold: float, maker_fee: float = 0.02) -> float:
    """
    Expected PnL per YES contract sold at yes_price_sold, with maker fee.
    P(YES wins) ≈ market implied probability = yes_price_sold
    P(YES loses) = 1 - yes_price_sold

    Net: if YES wins → you pay $1 - receive $yes_price_sold (from sale)
         if YES loses → NO pays $1, your NO purchase cost = (1 - yes_price_sold)
    """
    implied_prob = yes_price_sold  # market-implied P(YES)
    maker_proceeds = yes_price_sold * (1 - maker_fee)  # after fee
    return maker_proceeds - implied_prob  # positive = maker edge


# ── Simple maker PnL table ───────────────────────────────────────────

print("YES sold | Maker fee | PnL/contract | Label")
print("-" * 55)
for price in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    pnl = maker_pnl_per_contract(price, 0.02)
    flag = "✅ edge" if pnl > 0 else "❌ no edge"
    print(f"  {price:.2f}   |   2.0%   |   {pnl:+.4f}     | {flag}")
```

---

## Key Risks and Guard Rails

| Risk | Mitigation |
|---|---|
| Adverse selection (takers know more) | Only post in high-bias categories; limit to high-volume markets |
| Inventory risk (wrong direction) | Hard position limit per market; hedge with opposing NO if position > limit |
| Resolution uncertainty | Only trade markets with binary, publicly verifiable outcomes |
| Operational risk (cancel lag) | Kill switch — pause if spread widens > 200bps in 60s |
