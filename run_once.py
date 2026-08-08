"""
Single-cycle entry point for scheduled automation (e.g. GitHub Actions).

Unlike main.py (which loops forever with a 24h internal sleep -- fine for a
laptop left open, wrong for a scheduled job), this runs exactly one trading
cycle and exits. The daemon's own is_rebalance_day() logic still decides
whether that cycle actually places orders (every 10 trading days) or just
checks and does nothing -- so it's safe to schedule this to run daily.
"""

import asyncio
import os

from production_trading_daemon import QuantTradingDaemon, parse_universe, load_trained_universe


async def main():
    daemon = QuantTradingDaemon(
        tickers=parse_universe(load_trained_universe()),
        long_only=os.environ.get("LONG_ONLY", "false").lower() == "true",
        max_position_pct=float(os.environ.get("MAX_POSITION_PCT", "0.02")),
    )
    await daemon.run_trading_cycle()


if __name__ == "__main__":
    asyncio.run(main())
