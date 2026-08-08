import time
import numpy as np
import pandas as pd
from prometheus_client import start_http_server, Gauge, CollectorRegistry, REGISTRY

class PortfolioQuantExporter:
    """
    Exposes real-time portfolio risk and performance metrics
    to a Prometheus HTTP server (/metrics endpoint).
    """
    def __init__(self, port: int = 9108, risk_free_rate: float = 0.04):
        self.port = port
        self.risk_free_rate = risk_free_rate # Annualized risk-free rate (e.g., 4%)
        
        # --- Prometheus Gauge Metrics Setup ---
        # 1. Financial Performance Metrics
        self.metric_equity = Gauge(
            'portfolio_equity_usd', 
            'Current total portfolio NAV in USD',
            ['strategy_id']
        )
        self.metric_sharpe = Gauge(
            'portfolio_sharpe_ratio', 
            'Annualized Sharpe Ratio based on rolling returns window',
            ['strategy_id']
        )
        self.metric_max_drawdown = Gauge(
            'portfolio_max_drawdown_ratio', 
            'Maximum Drawdown ratio (0.0 to 1.0) from peak equity',
            ['strategy_id']
        )
        self.metric_win_rate = Gauge(
            'portfolio_win_rate_ratio', 
            'Proportion of profitable closed trades (0.0 to 1.0)',
            ['strategy_id']
        )
        
        # 2. Risk & Exposure Metrics
        self.metric_daily_volatility = Gauge(
            'portfolio_annualized_volatility', 
            'Annualized return volatility',
            ['strategy_id']
        )
        self.metric_unrealized_pnl = Gauge(
            'portfolio_unrealized_pnl_usd', 
            'Current total unrealized profit/loss in USD',
            ['strategy_id']
        )

    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculates the annualized Sharpe Ratio from periodic return series."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        rf_per_period = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = returns - rf_per_period
        
        # Annualized Sharpe = (Mean Excess Return / Std Dev) * sqrt(Periods)
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        return float(sharpe)

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculates Maximum Drawdown (MDD) as a decimal ratio."""
        if len(equity_curve) < 2:
            return 0.0
        
        running_max = equity_curve.cummax()
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = abs(drawdowns.min()) # Convert to positive ratio
        return float(max_dd)

    def update_metrics(self, strategy_id: str, equity_history: list[float], trades_pnl: list[float], unrealized_pnl: float):
        """
        Calculates risk statistics from state arrays and pushes 
        the updated values into Prometheus Gauges.
        """
        equity_series = pd.Series(equity_history)
        returns_series = equity_series.pct_change().dropna()
        
        # Compute key quant factors
        current_equity = equity_history[-1]
        sharpe_ratio = self.calculate_sharpe_ratio(returns_series)
        max_drawdown = self.calculate_max_drawdown(equity_series)
        
        # Win Rate
        closed_trades = np.array(trades_pnl)
        win_rate = float(np.sum(closed_trades > 0) / len(closed_trades)) if len(closed_trades) > 0 else 0.0
        
        # Annualized Volatility
        annual_vol = float(returns_series.std() * np.sqrt(252)) if len(returns_series) > 1 else 0.0

        # --- Push values into Prometheus metrics ---
        self.metric_equity.labels(strategy_id=strategy_id).set(current_equity)
        self.metric_sharpe.labels(strategy_id=strategy_id).set(sharpe_ratio)
        self.metric_max_drawdown.labels(strategy_id=strategy_id).set(max_drawdown)
        self.metric_win_rate.labels(strategy_id=strategy_id).set(win_rate)
        self.metric_daily_volatility.labels(strategy_id=strategy_id).set(annual_vol)
        self.metric_unrealized_pnl.labels(strategy_id=strategy_id).set(unrealized_pnl)

    def start(self):
        """Starts HTTP server exposing /metrics on the configured port."""
        print(f"[+] Starting Prometheus Metrics Exporter on http://localhost:{self.port}/metrics")
        start_http_server(self.port)


# --- Simulation Runner / Loop ---
if __name__ == "__main__":
    exporter = PortfolioQuantExporter(port=9108, risk_free_rate=0.04)
    exporter.start()

    # Mock historical portfolio equity and PnL state
    strategy_name = "transformer_alpha_v1"
    simulated_equity = [1_000_000.0]
    simulated_pnl_history = []
    
    print("[+] Simulating live portfolio stream... (Press Ctrl+C to stop)")
    
    try:
        while True:
            # Simulate a live tick / cycle update
            last_equity = simulated_equity[-1]
            daily_pct_change = np.random.normal(loc=0.0005, scale=0.012) # +0.05% mean, 1.2% vol
            new_equity = max(100.0, last_equity * (1 + daily_pct_change))
            
            simulated_equity.append(new_equity)
            if len(simulated_equity) > 252: # Keep rolling 1-year window
                simulated_equity.pop(0)

            # Record simulated trade return
            simulated_pnl_history.append(new_equity - last_equity)
            unrealized_pnl = np.random.uniform(-2500, 5000)

            # Update metrics in Prometheus
            exporter.update_metrics(
                strategy_id=strategy_name,
                equity_history=simulated_equity,
                trades_pnl=simulated_pnl_history[-50:], # Last 50 trades
                unrealized_pnl=unrealized_pnl
            )

            time.sleep(2) # Scrape/update interval
            
    except KeyboardInterrupt:
        print("\n[-] Exporter shut down.")
