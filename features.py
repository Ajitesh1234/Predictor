"""
Feature Engineering Module -- Cross-Sectional Equity Ranking Model.

SCOPE DECISION (deliberate, not an oversight):
------------------------------------------------
This version uses ONLY price-derived features (momentum, volatility,
technical). Earlier diagnostics found fundamentals extraction returning
0% coverage across the ENTIRE dataset (a yfinance field-matching failure),
and confirmed the model's out-of-sample edge came entirely from price-based
features anyway. Rather than re-import a data source already proven broken,
this scopes down to what's actually validated.

Every feature is cross-sectionally rank-transformed (0-1 percentile) within
each date, so a stock's momentum/vol/etc. is measured relative to its peers
on that date -- this is what makes the model "cross-sectional."
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "mom_12_1", "mom_6_1", "mom_1m_reversal",
    "vol_60d", "vol_120d", "rolling_beta",
    "dist_52w_high", "rsi_14", "ma_50_200_cross",
]


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    grp = df.groupby("ticker")["close"]

    df["mom_12_1"] = grp.transform(lambda s: s.shift(21) / s.shift(252) - 1)
    df["mom_6_1"] = grp.transform(lambda s: s.shift(21) / s.shift(126) - 1)
    df["mom_1m_reversal"] = grp.transform(lambda s: s / s.shift(21) - 1)
    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()

    df["vol_60d"] = df.groupby("ticker")["daily_return"].transform(
        lambda s: s.rolling(60, min_periods=40).std() * np.sqrt(252)
    )
    df["vol_120d"] = df.groupby("ticker")["daily_return"].transform(
        lambda s: s.rolling(120, min_periods=80).std() * np.sqrt(252)
    )
    return df


def compute_rolling_beta(df: pd.DataFrame, benchmark_ticker: str = "SPY", window: int = 120) -> pd.DataFrame:
    """Requires daily_return (run compute_volatility_features first) and
    that benchmark_ticker is present in df's universe."""
    df = df.sort_values(["ticker", "date"]).copy()

    bench = df.loc[df["ticker"] == benchmark_ticker, ["date", "daily_return"]].rename(
        columns={"daily_return": "bench_return"}
    )
    merged = df.merge(bench, on="date", how="left")

    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["daily_return"].rolling(window, min_periods=int(window * 0.6)).cov(g["bench_return"])
        var = g["bench_return"].rolling(window, min_periods=int(window * 0.6)).var()
        return cov / var

    merged["rolling_beta"] = merged.groupby("ticker", group_keys=False).apply(_beta)
    return merged.drop(columns=["bench_return"])


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    df["rolling_52w_high"] = df.groupby("ticker")["close"].transform(
        lambda s: s.rolling(252, min_periods=100).max()
    )
    df["dist_52w_high"] = df["close"] / df["rolling_52w_high"] - 1

    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = df.groupby("ticker")["close"].transform(_rsi)

    ma50 = df.groupby("ticker")["close"].transform(lambda s: s.rolling(50, min_periods=30).mean())
    ma200 = df.groupby("ticker")["close"].transform(lambda s: s.rolling(200, min_periods=100).mean())
    df["ma_50_200_cross"] = (ma50 > ma200).astype(float)
    return df


def compute_liquidity_filter(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    df["dollar_volume"] = df["close"] * df["volume"]
    df["adtv_20d"] = df.groupby("ticker")["dollar_volume"].transform(
        lambda s: s.rolling(window, min_periods=10).mean()
    )
    return df


def cross_sectional_rank(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Adds <col>_cs = percentile rank (0-1) of that feature within each date."""
    df = df.copy()
    for col in feature_cols:
        df[f"{col}_cs"] = df.groupby("date")[col].rank(pct=True)
    return df


def build_feature_matrix(
    pit_dataset_path: str = "./data/pit_dataset.parquet",
    output_path: str = "./data/features_dataset.parquet",
    min_adtv_usd: float = 5_000_000,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    df = pd.read_parquet(pit_dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_rolling_beta(df, benchmark_ticker=benchmark_ticker)
    df = compute_technical_features(df)
    df = compute_liquidity_filter(df)

    df = df[df["adtv_20d"] >= min_adtv_usd].copy()
    df = cross_sectional_rank(df, FEATURE_COLUMNS)

    df.to_parquet(output_path, index=False)
    print(f"[+] Saved feature matrix: {len(df):,} rows, {df['ticker'].nunique()} tickers -> {output_path}")
    return df


if __name__ == "__main__":
    build_feature_matrix()
