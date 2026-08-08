"""
Data Pipeline Module for Cross-Sectional Equity Ranking Model.

SURVIVORSHIP BIAS WARNING:
--------------------------
This implementation uses current S&P 500 constituents fetched from Wikipedia.
Using current constituents for historical backtesting introduces severe survivorship
bias because companies that went bankrupt, merged, or were delisted during the
target window (e.g., Lehman Brothers, Enron, SVB) are excluded.

In a true institutional setup, you MUST replace this step with historical
point-in-time constituent lists (e.g., via Norgate Data, CRSP, or Compustat).

POINT-IN-TIME (PIT) HYGIENE:
----------------------------
Fundamental data is lagged by 60 calendar days relative to the fiscal period end date
(`available_date = period_end + 60 days`) to simulate real-world reporting delays
and eliminate lookahead bias.

FUNDAMENTALS COVERAGE WARNING:
-------------------------------
yfinance's quarterly_financials / quarterly_balance_sheet endpoints typically only
return the last ~4-5 reported quarters per ticker, NOT a full multi-year history.
This means most of the price history in this dataset will likely have NaN
fundamentals for older dates. The pipeline prints a coverage report at the end
so you can see exactly how bad this is before building factors on top of it.
"""

import io
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
REPORTING_LAG_DAYS = 60
OUTPUT_DIR = Path("./data")
OUTPUT_FILE = OUTPUT_DIR / "pit_dataset.parquet"
PRICE_BATCH_SIZE = 50       # tickers per yfinance batch download
BATCH_PAUSE_SECONDS = 2     # pause between batches to avoid rate limiting
MAX_RETRIES = 2             # retries per failed batch


def fetch_sp500_tickers() -> List[str]:
    """
    Fetches the current list of S&P 500 tickers from Wikipedia.
    Uses requests with a custom User-Agent header to avoid HTTP 403 Forbidden errors.

    Flags explicit survivorship bias limitation.
    """
    logger.warning(
        "CRITICAL LIMITATION: Fetching CURRENT S&P 500 constituents. "
        "This introduces survivorship bias into historical backtests."
    )
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Pass raw HTML text via StringIO to avoid a pandas FutureWarning
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]

        # Standardize ticker formatting (e.g., BRK.B -> BRK-B for yfinance)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Successfully fetched {len(tickers)} current S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 tickers: {str(e)}")
        raise


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    """Splits a list into chunks of a given size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def fetch_daily_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads daily OHLCV data for given tickers via yfinance, in small batches
    with retries. Batching avoids the single-large-request failures and
    rate-limiting that are common when pulling ~500 tickers in one call.
    """
    logger.info(
        f"Downloading daily price data for {len(tickers)} tickers "
        f"from {start_date} to {end_date}..."
    )

    all_records = []
    failed_tickers = []
    batches = _chunk_list(tickers, PRICE_BATCH_SIZE)

    for batch_num, batch in enumerate(batches, start=1):
        logger.info(f"Price batch {batch_num}/{len(batches)} ({len(batch)} tickers)...")

        data = None
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                data = yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                break
            except Exception as e:
                logger.warning(f"  Batch {batch_num} attempt {attempt} failed: {e}")
                time.sleep(BATCH_PAUSE_SECONDS)

        if data is None or data.empty:
            for t in batch:
                failed_tickers.append((t, "Batch download failed after retries"))
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = data.copy()
                else:
                    if ticker not in data.columns.levels[0]:
                        failed_tickers.append((ticker, "No data returned from API"))
                        continue
                    df = data[ticker].dropna(how="all").copy()

                if df.empty:
                    failed_tickers.append((ticker, "Empty price history"))
                    continue

                df = df.reset_index()
                df["ticker"] = ticker

                df = df.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })

                df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
                all_records.append(df)
            except Exception as e:
                failed_tickers.append((ticker, str(e)))

        time.sleep(BATCH_PAUSE_SECONDS)  # be polite between batches

    if failed_tickers:
        logger.warning(f"Failed to process {len(failed_tickers)} tickers:")
        for t, reason in failed_tickers[:15]:
            logger.warning(f"  - Dropped {t}: {reason}")
        if len(failed_tickers) > 15:
            logger.warning(f"  - ... and {len(failed_tickers) - 15} more.")

    if not all_records:
        raise ValueError("Critical Error: No price data was successfully downloaded.")

    price_df = pd.concat(all_records, ignore_index=True)
    price_df["date"] = pd.to_datetime(price_df["date"])

    logger.info(
        f"Successfully loaded price data: {len(price_df):,} rows "
        f"across {price_df['ticker'].nunique()} tickers."
    )
    return price_df


def safe_extract_metric(df: pd.DataFrame, possible_names: List[str]) -> pd.Series:
    """Helper to safely extract financial line items across varying yfinance row names."""
    for name in possible_names:
        for idx in df.index:
            if name.lower() in str(idx).lower():
                return df.loc[idx]
    return pd.Series(np.nan, index=df.columns if not df.empty else [])


def fetch_quarterly_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """
    Extracts quarterly fundamentals using yfinance endpoints and applies an
    explicit 60-day reporting lag.

    NOTE ON DATA SOURCES:
    ---------------------
    yfinance fundamental dates correspond to fiscal period end dates, NOT exact
    filing dates. For institutional builds, substitute this with SEC EDGAR /
    Compustat / Alpha Vantage, which supply the exact `filing_date`
    (10-Q / 10-K submission timestamp).

    NOTE ON EPS (FIXED):
    ---------------------
    EPS only falls back from Diluted EPS to Basic EPS. It does NOT fall back to
    Net Income. Net income is a raw dollar figure (often hundreds of millions),
    not a per-share number, and silently substituting it would corrupt any
    downstream EPS-based feature. A missing EPS is left as NaN rather than
    filled with a value in the wrong unit.
    """
    logger.info("Extracting quarterly fundamentals and applying 60-day reporting lag...")
    fundamental_records = []
    failed_fund_tickers = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            logger.info(f"Processing fundamentals: {i + 1}/{len(tickers)} tickers complete...")

        try:
            t = yf.Ticker(ticker)
            fin = t.quarterly_financials
            bs = t.quarterly_balance_sheet

            if fin is None or fin.empty or bs is None or bs.empty:
                failed_fund_tickers.append((ticker, "Missing financial or balance sheet statements"))
                continue

            common_dates = fin.columns.intersection(bs.columns)
            if common_dates.empty:
                failed_fund_tickers.append((ticker, "No overlapping quarter dates between statements"))
                continue

            rev = safe_extract_metric(fin, ["Total Revenue", "Operating Revenue"])
            eps = safe_extract_metric(fin, ["Diluted EPS", "Basic EPS"])  # FIX: no Net Income fallback
            gross_profit = safe_extract_metric(fin, ["Gross Profit"])
            net_income = safe_extract_metric(fin, ["Net Income", "Net Income Common Stockholders"])

            equity = safe_extract_metric(
                bs, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"]
            )
            debt = safe_extract_metric(bs, ["Total Debt", "Long Term Debt"])

            for q_date in common_dates:
                r_val = rev.get(q_date, np.nan)
                e_val = eps.get(q_date, np.nan)
                gp_val = gross_profit.get(q_date, np.nan)
                ni_val = net_income.get(q_date, np.nan)
                eq_val = equity.get(q_date, np.nan)
                d_val = debt.get(q_date, np.nan)

                roe = (
                    (ni_val / eq_val)
                    if (pd.notna(ni_val) and pd.notna(eq_val) and eq_val != 0)
                    else np.nan
                )
                gross_margin = (
                    (gp_val / r_val)
                    if (pd.notna(gp_val) and pd.notna(r_val) and r_val != 0)
                    else np.nan
                )

                period_end = pd.to_datetime(q_date)
                # CRITICAL: Point-In-Time Availability Date
                available_date = period_end + pd.Timedelta(days=REPORTING_LAG_DAYS)

                fundamental_records.append({
                    "ticker": ticker,
                    "fiscal_period_end": period_end,
                    "available_date": available_date,
                    "revenue": r_val,
                    "eps": e_val,
                    "book_value": eq_val,
                    "total_debt": d_val,
                    "roe": roe,
                    "gross_margin": gross_margin
                })

        except Exception as e:
            failed_fund_tickers.append((ticker, str(e)))

    if failed_fund_tickers:
        logger.warning(f"Fundamental extraction warnings/drops ({len(failed_fund_tickers)} tickers):")
        for t, reason in failed_fund_tickers[:10]:
            logger.warning(f"  - Dropped {t}: {reason}")
        if len(failed_fund_tickers) > 10:
            logger.warning(f"  - ... and {len(failed_fund_tickers) - 10} more.")

    fund_df = pd.DataFrame(fundamental_records)
    if fund_df.empty:
        raise ValueError("Critical Error: No fundamental data was successfully extracted.")

    logger.info(f"Extracted {len(fund_df):,} fundamental quarterly records.")
    return fund_df


def merge_pit_data(price_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges prices and fundamentals using `pd.merge_asof` to guarantee
    zero lookahead bias. Asserts Point-In-Time integrity.
    """
    logger.info("Executing Point-in-Time (PIT) merge via `merge_asof`...")

    price_df = price_df.sort_values("date").reset_index(drop=True)
    fund_df = fund_df.sort_values("available_date").reset_index(drop=True)

    merged_df = pd.merge_asof(
        price_df,
        fund_df,
        left_on="date",
        right_on="available_date",
        by="ticker",
        direction="backward",
    )

    merged_df = merged_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    has_fundamentals = merged_df["available_date"].notna()
    violating_rows = merged_df[has_fundamentals & (merged_df["date"] < merged_df["available_date"])]

    assert len(violating_rows) == 0, (
        f"CRITICAL ERROR: Point-in-time leakage detected! "
        f"Found {len(violating_rows)} rows where trade date < available_date."
    )

    logger.info("ASSERTION PASSED: Zero lookahead leakage detected in fundamental merge.")
    return merged_df


def check_fundamentals_coverage(df: pd.DataFrame) -> None:
    """
    Prints how much of the dataset actually has fundamental data filled in,
    broken down by year, so you know up front whether Phase 2's value/quality
    factors will have enough real data to work with.
    """
    if df["revenue"].notna().sum() == 0:
        logger.warning("No fundamental data present at all in the merged dataset.")
        return

    earliest = df.loc[df["revenue"].notna(), "fiscal_period_end"].min()
    latest = df.loc[df["revenue"].notna(), "fiscal_period_end"].max()
    logger.info(f"Fundamentals coverage range: {earliest.date()} to {latest.date()}")

    df = df.copy()
    df["year"] = df["date"].dt.year
    coverage_by_year = df.groupby("year")["revenue"].apply(lambda s: s.notna().mean())
    logger.info("Fraction of rows with non-null revenue, by year:")
    for year, frac in coverage_by_year.items():
        logger.info(f"  {year}: {frac:.1%}")


def main():
    """Main execution function for Phase 1 data pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = fetch_sp500_tickers()
    price_df = fetch_daily_prices(tickers, start_date=START_DATE, end_date=END_DATE)
    fund_df = fetch_quarterly_fundamentals(tickers)
    pit_df = merge_pit_data(price_df, fund_df)

    check_fundamentals_coverage(pit_df)

    pit_df.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")
    logger.info(f"Pipeline complete. Saved PIT dataset to '{OUTPUT_FILE}' ({len(pit_df):,} rows).")


if __name__ == "__main__":
    main()
