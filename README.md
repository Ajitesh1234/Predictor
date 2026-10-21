# Predictor: Cross-Sectional Equity Ranking with Automated Paper Trading

> Educational research project. Not financial advice. All trading is on an
> Alpaca **paper** account (simulated money), and the execution code refuses
> to run against any endpoint that is not the paper API.

## What it does

Every 10 trading days the system scores about 500 large US stocks with a
LightGBM ranking model, goes long the top-ranked decile (50 stocks) and short
the bottom-ranked decile (50 stocks), and places the orders through the
Alpaca paper-trading API. A GitHub Actions workflow runs it on a weekday
schedule, so it operates without a laptop or server of mine switched on.

## Honest summary

The goal was not to "predict the stock market". It was to build the pipeline
the way quantitative research is done, and to test rigorously whether the
signal is real.

- The model shows a **small but real ranking signal** (Information Coefficient
  of about 0.016), concentrated almost entirely in the top-ranked stocks.
- Once market exposure was hedged out, the edge was **not statistically
  significant** (alpha t-statistic 0.76).
- Several early results looked much better than they were because of bugs in
  my own analysis. Finding and fixing them is documented below.

## Backtest results

Out-of-sample predictions from purged walk-forward validation, 10-day
rebalance, long top decile / short bottom decile, 10 bps cost per trade.

| | v1 (first build) | v2 (rebuild, currently deployed) |
|---|---|---|
| Mean IC | 0.0164 | 0.0155 |
| IC-IR | 0.083 | 0.080 |
| Raw long-short Sharpe | 0.89 | 0.67 |
| Raw annualized return | 14.9% | 7.5% |
| Max drawdown | about -20% | -16% |

Average 10-day forward return by predicted decile (v2, 1 = bottom, 10 = top):

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|
| 0.51% | 0.50% | 0.51% | 0.51% | 0.62% | 0.65% | 0.64% | 0.75% | 0.95% | 1.22% |

Deciles 1 to 5 are essentially indistinguishable. The model separates the top
of the ranking from the rest but cannot reliably identify the worst stocks.

Market-exposure analysis (run on v1, 228 independent 10-day periods):

| Portfolio | Beta to SPY | Annualized alpha | Alpha t-stat | Sharpe |
|---|---|---|---|---|
| Raw long-short | 0.24 | 13.3% | 2.24 | 0.89 |
| Beta-neutral | -0.08 | 3.4% | 0.76 | 0.13 |

Caveats on these numbers:
- The hedge and regression scripts from v1 are **not included in this
  repository**, so the table above is reported from the original runs and
  cannot be reproduced from this repo.
- The hedge analysis was not repeated on v2, which has a near-identical IC
  and decile profile.
- In v1, IC was negative in 2018, 2022 and 2026, so the edge is
  regime-dependent.

## Live paper-trading record

Account equity recorded by the system at each rebalance (from
`logs/rebalance_cycles.jsonl`). This is not a daily series.

| Rebalance date | Equity | Orders skipped |
|---|---|---|
| 2026-08-07 | $100,000 | 0 |
| 2026-08-21 | $102,953 | 1 (ES cannot be shorted) |
| 2026-09-04 | $103,515 | 1 (BRK-B symbol not found) |
| 2026-09-18 | $106,493 | 0 |
| 2026-10-02 | $112,966 | 0 |
| 2026-10-16 | $104,206 | 0 |

Equity fell about 7.8% between the last two rebalances. Six cycles is far too
little data to say whether the live results reflect skill, luck, or market
exposure.

## Method

**Universe:** 500 tickers in `data/universe.txt` (the S&P 500 constituents
that survived the data and liquidity filters, plus SPY as the market
benchmark).

**Features (9, all price-derived):** 12-1 month momentum, 6-1 month momentum,
1-month reversal, 60-day and 120-day volatility, 120-day rolling beta vs SPY,
distance from the 52-week high, RSI(14), and a 50/200-day moving-average
crossover. Each feature is converted to a cross-sectional percentile rank
within each date, so stocks are compared with each other, not with their own
history. A liquidity filter drops stocks under $5M average daily dollar
volume.

**Label:** 10-day forward return, converted to a percentile rank within each
date, then binned into 5 relevance levels for the ranker.

**Model:** LightGBM `LGBMRanker`, LambdaRank objective, one query group per
date (200 trees, learning rate 0.05, 31 leaves, seed 42).

**Validation:** expanding-window walk-forward cross-validation with
**purging** (training samples whose 10-day label window overlaps the test
period are removed) and a 10-day **embargo** after each test fold. 5 splits
are configured; the first has no prior history, so 4 folds are evaluated.

**Evaluation:** daily Spearman Information Coefficient, decile table, and a
cost-adjusted long-short backtest sampled every 10th trading day.

**Deployed model:** a frozen file, `models/best_model.pkl` (the final
walk-forward fold). It is not retrained automatically; retraining is a
deliberate manual step. The price data used for training ends in
December 2024, so all live trading happens after the data the model saw.

## Repository layout

| File | Purpose |
|---|---|
| `Pipeline.py` | Downloads 2015-2025 prices and builds the base dataset |
| `patch_add_spy.py` | Adds SPY (needed for rolling beta; `Pipeline.py` does not fetch it) |
| `features.py` | Feature engineering and cross-sectional ranking |
| `cv_splitter.py` | Purged walk-forward splitter and label construction |
| `train_model.py` | Trains the ranker, saves the model and out-of-sample predictions |
| `evaluate.py` | IC, decile table and cost-adjusted backtest |
| `extract_universe.py` | Writes `data/universe.txt` from the feature dataset |
| `production_trading_daemon.py` | Live feature building, scoring and portfolio construction |
| `execution_engine.py` | Alpaca paper API client with the paper-only safety check |
| `run_once.py` | Single-cycle entry point used by the automation |
| `main.py` | Local looping entry point (not used by the automation) |
| `.github/workflows/daily-trading.yml` | Weekday schedule |
| `logs/` | Rebalance state and per-cycle records, committed by the workflow |
| `models/`, `data/universe.txt` | The deployed model and ticker list |

**Not part of the validated pipeline or the automated run** (nothing in the
live path imports them): `Implementation.py` (an earlier classifier that was
superseded), `deep_temporal_attention_network.py`, `finbert_nlp_stream.py`,
`risk_engine.py`, `diagnose.py`. `portfolio_exporter.py` is only used by
`main.py`. The transformer and FinBERT files also need `torch`,
`transformers` and `edgartools`, which are not in `requirements.txt`.

## Reproducing the research

Large data files (`data/*.parquet`) are gitignored and regenerated locally.
The workflow uses Python 3.11.

```
pip install -r requirements.txt
python Pipeline.py            # download prices
python patch_add_spy.py       # add the SPY benchmark
python features.py
python train_model.py
python evaluate.py
python extract_universe.py
```

`lightgbm` and `scikit-learn` are pinned in `requirements.txt`; the saved
model may fail to load under other versions.

## Running the live system

Set these as environment variables locally, or as repository secrets on
GitHub:

```
ALPACA_API_KEY
ALPACA_SECRET_KEY
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

`run_once.py` runs one cycle. The workflow triggers it Monday to Friday at
14:00 UTC (GitHub often starts runs hours late). Each run checks how many
trading days have passed since the last rebalance (stored in
`logs/rebalance_state.json`) and only trades once 10 have passed. Every
rebalance is appended to `logs/rebalance_cycles.jsonl`, and the workflow
commits the logs back to the repository. If more than 15% of the universe
fails to download, the cycle refuses to trade.

## Bugs found and fixed

- **Fundamentals silently empty.** The value and quality features had 0%
  real data across the whole history because of a field-matching failure
  against the data source. I dropped them instead of keeping a broken feature.
- **Inflated statistical significance.** A regression on every stock-day
  instead of one return per 10-day period gave t = 7.46 next to R-squared of
  0.0009, which is impossible. Fixed to use independent periods.
- **Backtest compounding overlapping windows.** Daily sampling of 10-day
  returns produced a -79% drawdown next to an 11% volatility. Fixed by
  sampling every 10th trading day.
- **Unbounded leverage in beta-neutral sizing.** Dividing by near-zero betas
  produced a -6000% drawdown. Fixed with a beta floor and a leverage cap.
- **Missing benchmark.** SPY was not in the universe, so rolling beta was NaN
  for every row and the training set was empty.
- **Live and research features out of sync.** The live code first used 7
  different features on 10 tickers while the model expected 9 features ranked
  across about 500. It now imports the feature functions from `features.py`.
- **One rejected order crashed the whole cycle.** A stock that could not be
  shorted aborted all remaining orders. Each order is now handled
  independently and failures are logged.

## Limitations

- **Survivorship bias.** The backtest uses today's S&P 500 list, not
  point-in-time membership, so past performance is flattered.
- **Price-only features.** No fundamentals, news or alternative data.
- **Weak short leg.** Bottom-decile stocks are not reliably worse than
  average.
- **Regime dependence.** The signal has negative years.
- **Out-of-sample reuse.** The same out-of-sample period was inspected many
  times while debugging, so a truly untouched holdout is still needed.
- **Sizing mismatch.** The backtest splits capital 50/50 between longs and
  shorts. The live system holds 50 names per side at 2% each (100% long and
  100% short), so live returns are about twice as volatile as the backtest.
- **SPY is in the tradable universe.** It is needed as the beta benchmark but
  is also ranked like a stock, and it was shorted at 2% in four of the six
  rebalances so far. It should be excluded from trading.
- **Symbol mismatch.** Yahoo uses `BRK-B` while Alpaca uses `BRK.B`, so that
  stock cannot be traded; the order is skipped and logged.
- **Paper fills.** Paper trading does not capture real slippage, borrow costs
  or market impact. The backtest assumes a flat 10 bps per trade.

## Possible next steps

- Exclude SPY from the tradable universe and fix the `BRK-B` symbol mapping
- Fundamentals from SEC EDGAR, validated as a separate v2 before any change
  to the live model
- Point-in-time universe data
- Long-only versus long-short comparison
- Aligning live position sizing with the backtest
- Re-creating the hedge and regression diagnostics inside this repository
- A fresh holdout period evaluated once
