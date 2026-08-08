"""
Purged Walk-Forward Cross-Validation Splitter for Cross-Sectional Equity Ranking.

Implements purge + embargo methodology (Lopez de Prado, "Advances in
Financial Machine Learning") adapted for a fixed-horizon forward-return label.

WHY THIS EXISTS:
-----------------
If today's label uses the next N trading days' return, any training sample
whose label window overlaps the test period has effectively "seen" some of
the test period's price action. This splitter removes (purges) those
samples, and additionally embargoes a buffer of dates immediately after
EVERY earlier test fold before training resumes, to reduce leakage from
serial correlation. This is stateless (no hidden attributes carried between
calls to .split()) so it's safe to call multiple times.
"""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class PurgedWalkForwardSplit:
    """
    Time-ordered walk-forward cross-validator with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    label_horizon_days : int
        Number of trading days the forward-return label looks ahead. Any
        training date whose label window would overlap the test fold's
        start is purged.
    embargo_days : int
        Additional trading days immediately after EVERY earlier test fold
        excluded from later folds' training data.
    """

    def __init__(self, n_splits: int = 5, label_horizon_days: int = 10, embargo_days: int = 10):
        self.n_splits = n_splits
        self.label_horizon_days = label_horizon_days
        self.embargo_days = embargo_days

    def split(self, dates: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        dates = pd.Series(dates).reset_index(drop=True)
        unique_dates = np.sort(dates.unique())
        n_dates = len(unique_dates)

        if n_dates < self.n_splits * 2:
            raise ValueError(f"Not enough unique dates ({n_dates}) for {self.n_splits} splits.")

        fold_boundaries = np.linspace(0, n_dates, self.n_splits + 1, dtype=int)
        test_fold_ranges = [(fold_boundaries[i], fold_boundaries[i + 1]) for i in range(self.n_splits)]

        for i, (test_start_pos, test_end_pos) in enumerate(test_fold_ranges):
            if test_start_pos == 0:
                continue  # first fold has no prior training history

            test_dates = unique_dates[test_start_pos:test_end_pos]

            train_mask = np.zeros(n_dates, dtype=bool)
            train_mask[:test_start_pos] = True

            # PURGE: drop training dates whose label window would reach
            # into this test fold's start.
            purge_cutoff_pos = max(test_start_pos - self.label_horizon_days, 0)
            train_mask[purge_cutoff_pos:test_start_pos] = False

            # EMBARGO: drop embargo_days after EVERY earlier test fold.
            for j in range(i):
                prior_test_end_pos = test_fold_ranges[j][1]
                embargo_hi = min(prior_test_end_pos + self.embargo_days, n_dates)
                train_mask[prior_test_end_pos:embargo_hi] = False

            train_dates = unique_dates[train_mask]
            if len(train_dates) == 0:
                continue

            train_idx = np.where(dates.isin(train_dates))[0]
            test_idx = np.where(dates.isin(test_dates))[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def summary(self, dates: pd.Series) -> pd.DataFrame:
        """Human-readable per-fold date ranges, for visually verifying no leakage."""
        dates = pd.Series(dates).reset_index(drop=True)
        rows = []
        for fold_num, (train_idx, test_idx) in enumerate(self.split(dates), start=1):
            train_dates = dates.iloc[train_idx]
            test_dates = dates.iloc[test_idx]
            rows.append({
                "fold": fold_num,
                "train_start": train_dates.min(), "train_end": train_dates.max(),
                "n_train_dates": train_dates.nunique(),
                "test_start": test_dates.min(), "test_end": test_dates.max(),
                "n_test_dates": test_dates.nunique(),
            })
        return pd.DataFrame(rows)


def build_forward_return_labels(df: pd.DataFrame, price_col: str = "close", horizon_days: int = 10) -> pd.DataFrame:
    """
    Adds 'actual_forward_return' (raw forward return over horizon_days) and
    'label_rank' (cross-sectional percentile rank 0-1 per date) to df.
    Rows within horizon_days of a ticker's last available date will have
    NaN forward returns -- drop these before training, but they can still
    be used at inference time to score the most recent date.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    df["actual_forward_return"] = (
        df.groupby("ticker")[price_col].transform(lambda s: s.shift(-horizon_days) / s - 1)
    )
    df["label_rank"] = df.groupby("date")["actual_forward_return"].rank(pct=True)

    return df
