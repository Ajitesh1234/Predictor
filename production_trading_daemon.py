from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from execution_engine import AlpacaPaperExecutionEngine
from features import (
    compute_momentum_features,
    compute_volatility_features,
    compute_rolling_beta,
    compute_technical_features,
    compute_liquidity_filter,
    cross_sectional_rank,
    FEATURE_COLUMNS,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

VALIDATED_MODEL_PATH = Path("models") / "best_model.pkl"
UNIVERSE_PATH = Path("data") / "universe.txt"

# Single source of truth: derived directly from features.py, exact same
# order used in train_model.py. This cannot drift out of sync with the
# trained model the way a separately hardcoded list could.
VALIDATED_FEATURE_COLUMNS = [f"{c}_cs" for c in FEATURE_COLUMNS]

PRICE_BATCH_SIZE = 50
BATCH_PAUSE_SECONDS = 2
MAX_RETRIES = 2
MAX_MISSING_FRACTION = 0.15  # refuse to trade if more than 15% of universe fails to fetch


def load_trained_universe() -> list[str]:
    if not UNIVERSE_PATH.exists():
        raise SystemExit(
            f"{UNIVERSE_PATH} not found. Run extract_universe.py once "
            "(after train_model.py) to generate it."
        )
    universe = [line.strip() for line in UNIVERSE_PATH.read_text().splitlines() if line.strip()]
    print(f"[+] Loaded trained universe: {len(universe)} tickers from {UNIVERSE_PATH}")
    return universe


class QuantTradingDaemon:
    """
    Paper-only production daemon that trades the validated cross-sectional
    ranking model on its original 10-trading-day forward-return cadence.
    """

    def __init__(
        self,
        tickers: list[str],
        model_path: Path | str = VALIDATED_MODEL_PATH,
        rebalance_trading_days: int = 10,
        max_position_pct: float = 0.02,
        long_only: bool = False,
        log_path: Path | str = Path("logs") / "rebalance_cycles.jsonl",
        state_path: Path | str = Path("logs") / "rebalance_state.json",
    ):
        self.tickers = tickers
        self.model_path = Path(model_path)
        self.rebalance_trading_days = rebalance_trading_days
        self.max_position_pct = max_position_pct
        self.long_only = long_only
        self.log_path = Path(log_path)
        self.state_path = Path(state_path)
        self.feature_cols = VALIDATED_FEATURE_COLUMNS.copy()

        print("[+] Initializing PAPER-ONLY Quant Trading Daemon...")
        self.exec_engine = AlpacaPaperExecutionEngine(max_position_pct=self.max_position_pct)
        self.model = self._load_validated_model()

        mode = "LONG_ONLY" if self.long_only else "LONG_SHORT"
        print(f"[+] Portfolio construction mode: {mode}")
        print(f"[+] Rebalance cadence: every {self.rebalance_trading_days} trading days")
        print(f"[+] Universe size: {len(self.tickers)} tickers")
        print(f"[+] Validated feature columns: {', '.join(self.feature_cols)}")

    def _load_validated_model(self):
        model_path = self.model_path.resolve()
        if not model_path.exists():
            message = (
                f"Validated model artifact missing: {model_path}. "
                "Refusing to trade without models/best_model.pkl."
            )
            logger.critical(message)
            raise SystemExit(message)

        modified = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
        print(f"[+] Validated model path: {model_path}")
        print(f"[+] Validated model last modified UTC: {modified.isoformat()}")

        try:
            try:
                import joblib
                model = joblib.load(model_path)
            except ImportError:
                with model_path.open("rb") as model_file:
                    model = pickle.load(model_file)
        except Exception:
            logger.exception("Failed to load validated model from %s", model_path)
            raise

        if not hasattr(model, "predict_proba") and not hasattr(model, "predict"):
            message = f"Validated model lacks predict_proba/predict interface: {type(model)!r}"
            logger.critical(message)
            raise SystemExit(message)

        return model

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        with self.state_path.open("r", encoding="utf-8") as state_file:
            return json.load(state_file)

    def _save_state(self, state: dict):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)

    def _trading_days_since_last_rebalance(self, today: date) -> int | None:
        state = self._load_state()
        last_rebalance = state.get("last_rebalance_date")
        if not last_rebalance:
            return None

        last_date = date.fromisoformat(last_rebalance)
        if today <= last_date:
            return 0

        business_days = pd.bdate_range(last_date + timedelta(days=1), today)
        return len(business_days)

    def is_rebalance_day(self, today: date | None = None) -> bool:
        today = today or date.today()
        days_since = self._trading_days_since_last_rebalance(today)
        if days_since is None:
            print("[+] No previous rebalance date found; today is a rebalance day.")
            return True

        print(
            f"[+] Trading days since last rebalance: {days_since}/"
            f"{self.rebalance_trading_days}"
        )
        return days_since >= self.rebalance_trading_days

    def _download_live_prices(self) -> pd.DataFrame:
        """
        Fetches ~2 years of daily OHLCV for the full universe, batched with
        retries (same pattern as pipeline.py) to avoid rate-limit failures
        at this scale. Returns a long-format dataframe: date, ticker, close,
        volume -- ready to feed into features.py's functions.
        """
        batches = [self.tickers[i:i + PRICE_BATCH_SIZE] for i in range(0, len(self.tickers), PRICE_BATCH_SIZE)]
        all_rows = []
        failed_tickers = []

        for batch_num, batch in enumerate(batches, start=1):
            print(f"[+] Live price batch {batch_num}/{len(batches)} ({len(batch)} tickers)...")
            data = None
            for attempt in range(1, MAX_RETRIES + 2):
                try:
                    data = yf.download(
                        tickers=batch, period="2y", interval="1d",
                        group_by="ticker", auto_adjust=True, threads=True, progress=False,
                    )
                    break
                except Exception as e:
                    logger.warning("Batch %s attempt %s failed: %s", batch_num, attempt, e)
                    time.sleep(BATCH_PAUSE_SECONDS)

            if data is None or data.empty:
                failed_tickers.extend(batch)
                continue

            for ticker in batch:
                try:
                    df = data[ticker].copy() if len(batch) > 1 else data.copy()
                    df = df.dropna(subset=["Close"])
                    if df.empty:
                        failed_tickers.append(ticker)
                        continue
                    df = df.reset_index().rename(columns={
                        "Date": "date", "Close": "close", "Volume": "volume",
                    })
                    df["ticker"] = ticker
                    all_rows.append(df[["date", "ticker", "close", "volume"]])
                except Exception:
                    failed_tickers.append(ticker)

            time.sleep(BATCH_PAUSE_SECONDS)

        missing_fraction = len(failed_tickers) / max(len(self.tickers), 1)
        if failed_tickers:
            logger.warning("Failed to fetch %s/%s tickers: %s", len(failed_tickers), len(self.tickers), failed_tickers[:20])

        if missing_fraction > MAX_MISSING_FRACTION:
            raise RuntimeError(
                f"Too many tickers failed to fetch ({len(failed_tickers)}/{len(self.tickers)}, "
                f"{missing_fraction:.1%}). Refusing to trade on a degraded universe."
            )

        if not all_rows:
            raise RuntimeError("Live price fetch returned no usable data for any ticker.")

        combined = pd.concat(all_rows, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        return combined

    def _sync_fetch_feature_frame(self) -> pd.DataFrame:
        """
        Builds live features using the EXACT SAME functions as features.py,
        applied to freshly downloaded price history -- no separate/ad-hoc
        feature logic that could drift from what the model was trained on.
        """
        price_df = self._download_live_prices()

        df = compute_momentum_features(price_df)
        df = compute_volatility_features(df)
        df = compute_rolling_beta(df, benchmark_ticker="SPY")
        df = compute_technical_features(df)
        df = compute_liquidity_filter(df)
        df = cross_sectional_rank(df, FEATURE_COLUMNS)

        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].copy()

        required_cols = self.feature_cols + ["close"]
        before = len(latest)
        latest = latest.dropna(subset=required_cols)
        dropped = before - len(latest)
        if dropped:
            print(f"[+] Dropped {dropped} tickers with incomplete features on {latest_date.date()} "
                  f"(insufficient price history for rolling windows).")

        if latest.empty:
            raise RuntimeError(
                f"No tickers had complete features on {latest_date.date()} -- "
                "likely insufficient price history depth."
            )

        print(f"[+] Live feature frame: {len(latest)} tickers scored as of {latest_date.date()}")
        return latest.set_index("ticker")[required_cols]

    async def fetch_feature_frame(self) -> pd.DataFrame:
        return await asyncio.to_thread(self._sync_fetch_feature_frame)

    def rank_universe(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        features = feature_frame[self.feature_cols]
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(features)[:, 1]
        else:
            scores = self.model.predict(features)

        ranked = feature_frame.copy()
        ranked["alpha_score"] = scores
        ranked.sort_values("alpha_score", ascending=False, inplace=True)
        return ranked

    def construct_target_weights(self, ranked: pd.DataFrame) -> pd.Series:
        n_assets = len(ranked)
        if n_assets == 0:
            raise RuntimeError("Cannot construct portfolio from empty ranking.")

        decile_size = max(1, int(np.ceil(n_assets * 0.10)))
        target_weights = pd.Series(0.0, index=ranked.index)

        long_names = ranked.head(decile_size).index
        long_weight = min(self.max_position_pct, 1.0 / len(long_names))
        target_weights.loc[long_names] = long_weight

        if not self.long_only:
            short_names = ranked.tail(decile_size).index
            short_weight = -min(self.max_position_pct, 1.0 / len(short_names))
            target_weights.loc[short_names] = short_weight

        return target_weights

    def _log_rebalance_cycle(self, ranked, target_weights, orders, equity, skipped, reason):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "date": date.today().isoformat(),
            "model_path": str(self.model_path.resolve()),
            "model_last_modified_utc": datetime.fromtimestamp(
                self.model_path.resolve().stat().st_mtime, tz=timezone.utc,
            ).isoformat(),
            "long_only": self.long_only,
            "max_position_pct": self.max_position_pct,
            "skipped": skipped,
            "skip_reason": reason,
            "ranked_list": [
                {
                    "ticker": ticker,
                    "alpha_score": float(row["alpha_score"]),
                    "close": float(row["close"]),
                    "target_weight": float(target_weights.get(ticker, 0.0)),
                }
                for ticker, row in ranked.iterrows()
            ],
            "orders": orders,
            "paper_account_equity": equity,
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, sort_keys=True) + "\n")

    async def run_trading_cycle(self, force_rebalance: bool = False, dry_run: bool = False):
        print(f"\n[CYCLE START] {datetime.now(timezone.utc).isoformat()}")
        account = self.exec_engine.get_account()
        print(f"[+] PAPER account equity: ${account.equity:,.2f}")

        today = date.today()
        if not force_rebalance and not self.is_rebalance_day(today):
            print("[+] Not a scheduled rebalance day. No rankings or orders generated.")
            return {"skipped": True, "reason": "not_rebalance_day", "equity": account.equity}

        feature_frame = await self.fetch_feature_frame()
        ranked = self.rank_universe(feature_frame)
        target_weights = self.construct_target_weights(ranked)

        print("\n[VALIDATED MODEL RANKINGS] (top 10 / bottom 10 shown)")
        for ticker, row in pd.concat([ranked.head(10), ranked.tail(10)]).iterrows():
            print(f"  {ticker:<6} -> score={row['alpha_score']:.6f} target_weight={target_weights[ticker]:+.2%}")

        current_prices = ranked["close"].to_dict()
        orders = self.exec_engine.rebalance_portfolio(target_weights.to_dict(), current_prices, dry_run=dry_run)
        ending_account = self.exec_engine.get_account()
        self._log_rebalance_cycle(ranked, target_weights, orders, ending_account.equity, False, None)

        if not dry_run:
            self._save_state({"last_rebalance_date": today.isoformat()})

        return {"skipped": False, "orders": orders, "equity": ending_account.equity}

    async def start_daemon(self, daily_check_interval_seconds: int = 86_400):
        print(f"[+] Trading Daemon live. Daily rebalance checks every {daily_check_interval_seconds}s.")
        try:
            while True:
                await self.run_trading_cycle()
                await asyncio.sleep(daily_check_interval_seconds)
        except KeyboardInterrupt:
            print("\n[-] Daemon stopped cleanly by operator.")


def parse_universe(default: Iterable[str]) -> list[str]:
    raw_universe = os.environ.get("TICKER_UNIVERSE")
    if not raw_universe:
        return list(default)
    tickers = [ticker.strip().upper() for ticker in raw_universe.split(",") if ticker.strip()]
    if not tickers:
        raise SystemExit("TICKER_UNIVERSE was set but contained no tickers.")
    return tickers


if __name__ == "__main__":
    daemon = QuantTradingDaemon(
        tickers=parse_universe(load_trained_universe()),
        long_only=os.environ.get("LONG_ONLY", "false").lower() == "true",
        max_position_pct=float(os.environ.get("MAX_POSITION_PCT", "0.02")),
    )
    asyncio.run(daemon.run_trading_cycle(force_rebalance=True))
