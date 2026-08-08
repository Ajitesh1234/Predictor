"""
One-time patch: adds SPY price history to the existing pit_dataset.parquet.

WHY: pipeline.py only fetches S&P 500 constituent stocks, but features.py's
compute_rolling_beta() needs SPY (an index ETF, not a constituent) as the
benchmark. Without it, every row's rolling_beta comes out NaN, which wipes
out the entire dataset in train_model.py's dropna() step.

Run this once, then re-run features.py and train_model.py. No need to
redownload the other ~500 tickers.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

PIT_PATH = Path("./data/pit_dataset.parquet")
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

# Must match pit_dataset.parquet's schema exactly
FUNDAMENTAL_COLS = [
    "fiscal_period_end", "available_date", "revenue",
    "eps", "book_value", "total_debt", "roe", "gross_margin",
]


def main():
    df = pd.read_parquet(PIT_PATH)

    if "SPY" in df["ticker"].unique():
        print("[+] SPY already present in pit_dataset.parquet -- nothing to do.")
        print("[!] If train_model.py still shows 0 rows, the cause is something else -- report back.")
        return

    print("[+] SPY not found. Fetching SPY price history...")
    spy = yf.download("SPY", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if spy.empty:
        raise ValueError("SPY download returned no data -- check network/yfinance status before retrying.")

    # Handle yfinance sometimes returning MultiIndex columns even for one ticker
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]

    spy = spy.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    spy["ticker"] = "SPY"
    spy["date"] = pd.to_datetime(spy["date"])

    # SPY is an ETF -- it has no quarterly fundamentals in this schema.
    # Leave these as missing; merge/feature code already handles NaN fundamentals.
    for col in FUNDAMENTAL_COLS:
        spy[col] = pd.NA

    spy = spy[df.columns]  # exact column order match before concatenating

    combined = pd.concat([df, spy], ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    combined.to_parquet(PIT_PATH, index=False)

    print(f"[+] Added {len(spy):,} SPY rows. New total: {len(combined):,} rows.")
    print("[+] Now re-run: python features.py, then python train_model.py")


if __name__ == "__main__":
    main()
