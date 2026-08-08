import asyncio
import os
from portfolio_exporter import PortfolioQuantExporter
from production_trading_daemon import QuantTradingDaemon, parse_universe, load_trained_universe


async def main():
    # 1. Start Prometheus Metrics Server on Port 9108
    exporter = PortfolioQuantExporter(port=9108, risk_free_rate=0.04)
    exporter.start()

    # 2. Initialize and Run the Paper-Only Quant Trading Daemon
    default_universe = load_trained_universe()
    daemon = QuantTradingDaemon(
        tickers=parse_universe(default_universe),
        long_only=os.environ.get("LONG_ONLY", "false").lower() == "true",
        max_position_pct=float(os.environ.get("MAX_POSITION_PCT", "0.02")),
    )

    # Check rebalance eligibility once per day; actual orders only every 10 trading days.
    await daemon.start_daemon(daily_check_interval_seconds=86_400)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[-] System shut down cleanly.")
