#!/usr/bin/env python3
"""
Polymarket Copy Trading Bot (Rust) — Reference Pattern
The actual implementation is in Rust. This shows the config/setup pattern.
"""

# === Config.toml (target format) ===
CONFIG_EXAMPLE = """
[wallet]
private_key = "0x..."
rpc_url = "https://polygon-rpc.com"

[copy_trading]
# Comma-separated wallet addresses to follow
target_wallets = [
    "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",
    "0x..."
]
# Multiplier for position sizing (0.0 - 1.0)
trade_multiplier = 0.5
# Aggregation window in seconds
aggregation_window = 30

[risk]
max_daily_loss_usd = 100
max_position_usd = 50
dry_run = true

[monitoring]
poll_interval_ms = 500
websocket_fallback = true
"""

# === Build & Run Commands ===
BUILD_COMMANDS = """
# Build release binary
cargo build --release

# Dry run with paper trading
cargo run --release -- --dry-run

# Live trading
cargo run --release
"""

if __name__ == "__main__":
    print("=== Polymarket Copy Trading Bot (Rust) ===")
    print()
    print("Build & run (Rust):")
    print(BUILD_COMMANDS)
    print()
    print("Configuration:")
    print(CONFIG_EXAMPLE)
