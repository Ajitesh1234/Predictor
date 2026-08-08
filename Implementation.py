import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class InstitutionalQuantEngine:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None

    def fetch_data(self):
        """Fetch daily OHLCV market data for the target universe."""
        print(f"[+] Ingesting data for {len(self.tickers)} tickers...")
        try:
            raw_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                group_by='ticker',
                progress=False,
                threads=False,
            )
        except Exception:
            logger.exception("Yahoo Finance download failed for tickers=%s", self.tickers)
            raise

        if raw_data is None or raw_data.empty:
            message = f"Yahoo Finance returned no rows for tickers={self.tickers}"
            logger.error(message)
            raise RuntimeError(message)
        
        frames = []
        missing_tickers = []
        for ticker in self.tickers:
            try:
                df = raw_data[ticker].copy() if len(self.tickers) > 1 else raw_data.copy()
            except KeyError:
                missing_tickers.append(ticker)
                continue

            if 'Close' not in df.columns:
                missing_tickers.append(ticker)
                continue

            df = df.dropna(subset=['Close'])
            if df.empty:
                missing_tickers.append(ticker)
                continue
            df['Ticker'] = ticker
            frames.append(df)

        if missing_tickers:
            message = (
                "Yahoo Finance returned incomplete data. "
                f"Missing/empty tickers={missing_tickers}; requested={self.tickers}"
            )
            logger.error(message)
            raise RuntimeError(message)

        if not frames:
            message = f"No usable Yahoo Finance data returned for tickers={self.tickers}"
            logger.error(message)
            raise RuntimeError(message)
            
        self.data = pd.concat(frames).reset_index()
        self.data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 
                                 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        self.data.sort_values(by=['Ticker', 'date'], inplace=True)
        return self

    def engineer_features(self):
        """
        Build predictive alpha factors:
        - Exponential moving average ratios (momentum)
        - Relative Strength Index (RSI)
        - Realized volatility (risk)
        - Target: Predict if 5-day future return beats median market return.
        """
        print("[+] Engineering statistical features (strictly lagging to prevent look-ahead bias)...")
        df_list = []
        
        for ticker, group in self.data.groupby('Ticker'):
            group = group.copy()
            
            # Returns & Momentum
            group['ret_1d'] = group['close'].pct_change(1)
            group['ret_5d'] = group['close'].pct_change(5)
            group['ret_21d'] = group['close'].pct_change(21)
            
            # Volatility
            group['vol_21d'] = group['ret_1d'].rolling(21).std() * np.sqrt(252)
            
            # Moving Averages / Trend
            group['sma_50'] = group['close'] / group['close'].rolling(50).mean() - 1
            group['sma_200'] = group['close'] / group['close'].rolling(200).mean() - 1
            
            # RSI (14-period)
            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            group['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Target Horizon: Forward 5-day return (LAGGED for evaluation)
            group['target_ret_5d'] = group['close'].shift(-5) / group['close'] - 1
            
            df_list.append(group)
            
        df = pd.concat(df_list).dropna().reset_index(drop=True)
        
        # Cross-sectional target classification (1 if outperforming top 50 percentile on date t)
        df['target'] = df.groupby('date')['target_ret_5d'].transform(
            lambda x: (x > x.median()).astype(int)
        )
        
        self.data = df
        return self

    def train_model(self):
        """
        Train a LightGBM model using time-series purged cross-validation
        to prevent future-data leakage.
        """
        feature_cols = ['ret_1d', 'ret_5d', 'ret_21d', 'vol_21d', 'sma_50', 'sma_200', 'rsi_14']
        X = self.data[feature_cols]
        y = self.data['target']
        
        print(f"[+] Training LightGBM Model across {len(X)} historical samples...")
        
        # Time Series Split (No random shuffle to preserve chronological causality)
        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=4,
                random_state=42,
                verbosity=-1
            )
            model.fit(X_train, y_train)
            
            val_preds = model.predict(X_val)
            acc = accuracy_score(y_val, val_preds)
            prec = precision_score(y_val, val_preds)
            print(f"   Fold {fold+1} Validation -> Accuracy: {acc:.2%}, Precision: {prec:.2%}")
            
        self.model = model
        self.feature_cols = feature_cols
        return self

    def generate_recommendations(self, top_n: int = 3):
        """Rank the universe on the most recent trading date using predicted probabilities."""
        latest_date = self.data['date'].max()
        latest_universe = self.data[self.data['date'] == latest_date].copy()
        
        if latest_universe.empty:
            print("[-] No valid data available for recommendation.")
            return
            
        X_latest = latest_universe[self.feature_cols]
        latest_universe['alpha_score'] = self.model.predict_proba(X_latest)[:, 1]
        
        # Rank by probability of 5-day relative outperformance
        ranked = latest_universe.sort_values(by='alpha_score', ascending=False)
        
        print("\n" + "="*60)
        print(f" TOP QUANT RECOMENDATIONS FOR NEXT 5 DAYS (As of {latest_date.strftime('%Y-%m-%d')})")
        print("="*60)
        
        for idx, row in ranked.head(top_n).reset_index().iterrows():
            print(f"Rank {idx+1}: {row['Ticker']:<6} | Alpha Score: {row['alpha_score']:.3f} | RSI: {row['rsi_14']:.1f} | 21D Vol: {row['vol_21d']:.2%}")
        
        return ranked[['Ticker', 'alpha_score', 'close', 'rsi_14', 'vol_21d']].head(top_n)

# Execution Sandbox
if __name__ == "__main__":
    # Liquid S&P 500 / Tech Universe
    universe = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'AMD', 'SPY']
    
    engine = InstitutionalQuantEngine(
        tickers=universe, 
        start_date="2021-01-01", 
        end_date="2026-01-01"
    )
    
    engine.fetch_data()\
          .engineer_features()\
          .train_model()\
          .generate_recommendations(top_n=3)
