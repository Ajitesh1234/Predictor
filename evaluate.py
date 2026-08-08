"""
Evaluation -- IC, Decile Spread, and Backtest for the Cross-Sectional
Ranking Model's Out-of-Sample Predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OOS_PREDICTIONS_PATH = Path("./data/oos_predictions.parquet")

TRANSACTION_COST_BPS = 10
REBALANCE_DAYS = 10


def compute_ic(df: pd.DataFrame) -> pd.Series:
    def _daily_ic(g: pd.DataFrame):
        if g["predicted_score"].nunique() < 2 or g["actual_forward_return"].nunique() < 2:
            return np.nan
        corr, _ = spearmanr(g["predicted_score"], g["actual_forward_return"])
        return corr

    return df.groupby("date").apply(_daily_ic).dropna()


def compute_decile_table(df: pd.DataFrame, n_deciles: int = 10) -> pd.Series:
    df = df.copy()
    df["decile"] = df.groupby("date")["predicted_score"].transform(
        lambda s: pd.qcut(s, n_deciles, labels=False, duplicates="drop") + 1
    )
    return df.groupby("decile")["actual_forward_return"].mean()


def backtest_long_short(df, n_deciles=10, cost_bps=TRANSACTION_COST_BPS, rebalance_days=REBALANCE_DAYS):
    df = df.copy()
    df["decile"] = df.groupby("date")["predicted_score"].transform(
        lambda s: pd.qcut(s, n_deciles, labels=False, duplicates="drop") + 1
    )

    unique_dates = np.sort(df["date"].unique())
    rebalance_dates = unique_dates[::rebalance_days]  # every 10th trading date ONLY

    period_returns = []
    prev_long_set, prev_short_set = set(), set()

    for date in rebalance_dates:
        g = df[df["date"] == date]
        if g.empty:
            continue
        long_tickers = set(g.loc[g["decile"] == n_deciles, "ticker"])
        short_tickers = set(g.loc[g["decile"] == 1, "ticker"])

        long_ret = g.loc[g["ticker"].isin(long_tickers), "actual_forward_return"].mean()
        short_ret = g.loc[g["ticker"].isin(short_tickers), "actual_forward_return"].mean()
        gross_ret = (long_ret - short_ret) / 2

        turnover = len(long_tickers.symmetric_difference(prev_long_set)) + \
            len(short_tickers.symmetric_difference(prev_short_set))
        total_names = max(len(long_tickers) + len(short_tickers), 1)
        cost = (turnover / total_names) * (cost_bps / 10_000)

        period_returns.append(gross_ret - cost)
        prev_long_set, prev_short_set = long_tickers, short_tickers

    period_returns = pd.Series(period_returns).dropna()
    periods_per_year = 252 / rebalance_days

    ann_return = (1 + period_returns).prod() ** (periods_per_year / len(period_returns)) - 1
    ann_vol = period_returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + period_returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.min(),
        "n_periods": len(period_returns),
    }


def main():
    df = pd.read_parquet(OOS_PREDICTIONS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[+] Loaded {len(df):,} OOS predictions across {df['date'].nunique():,} dates.")

    ic_series = compute_ic(df)
    print(f"\nMean IC : {ic_series.mean():.4f}")
    print(f"IC Std  : {ic_series.std():.4f}")
    print(f"IC-IR   : {ic_series.mean() / ic_series.std():.4f}")

    print("\nAverage return by decile (1=bottom, 10=top):")
    print(compute_decile_table(df))

    print("\nBacktest (long-short decile 10/1, cost-adjusted):")
    for k, v in backtest_long_short(df).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
