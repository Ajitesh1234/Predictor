"""
Model Training -- Cross-Sectional Equity Ranking Model.

Trains a LightGBM LambdaRank model using purged walk-forward cross-
validation, and saves both the final model and the full out-of-sample
prediction set (needed for evaluate.py). This is a from-scratch rebuild
matching the exact methodology validated earlier in this project: NOT the
InstitutionalQuantEngine classifier that was mistakenly trained on 10
tickers with a plain TimeSeriesSplit -- do not reuse that model file.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from cv_splitter import PurgedWalkForwardSplit, build_forward_return_labels
from features import FEATURE_COLUMNS

FEATURES_PATH = Path("./data/features_dataset.parquet")
MODEL_PATH = Path("./models/best_model.pkl")
OOS_PREDICTIONS_PATH = Path("./data/oos_predictions.parquet")

LABEL_HORIZON_DAYS = 10
N_SPLITS = 5
EMBARGO_DAYS = 10

CS_FEATURE_COLUMNS = [f"{c}_cs" for c in FEATURE_COLUMNS]


def main():
    print(f"[+] Loading feature matrix from {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print("[+] Building forward-return labels...")
    df = build_forward_return_labels(df, price_col="close", horizon_days=LABEL_HORIZON_DAYS)

    required_cols = ["actual_forward_return", "label_rank"] + CS_FEATURE_COLUMNS
    labeled_df = df.dropna(subset=required_cols).copy()
    labeled_df = labeled_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"[+] {len(labeled_df):,} labeled rows across {labeled_df['date'].nunique():,} dates.")

    splitter = PurgedWalkForwardSplit(
        n_splits=N_SPLITS, label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS
    )

    all_oos_predictions = []
    final_model = None

    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(labeled_df["date"]), start=1):
        train_df = labeled_df.iloc[train_idx].copy()
        test_df = labeled_df.iloc[test_idx].copy()

        print(
            f"[fold {fold_num}] train: {train_df['date'].min().date()} -> {train_df['date'].max().date()} "
            f"({len(train_df):,} rows) | test: {test_df['date'].min().date()} -> {test_df['date'].max().date()} "
            f"({len(test_df):,} rows)"
        )

        # LGBMRanker needs an integer relevance label plus a `group` array
        # giving row-count-per-query (one query = one date).
        train_df["relevance"] = pd.qcut(train_df["label_rank"], 5, labels=False, duplicates="drop")
        test_df["relevance"] = pd.qcut(test_df["label_rank"], 5, labels=False, duplicates="drop")

        train_group = train_df.groupby("date").size().values

        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            random_state=42,
        )
        model.fit(train_df[CS_FEATURE_COLUMNS], train_df["relevance"], group=train_group)

        test_df["predicted_score"] = model.predict(test_df[CS_FEATURE_COLUMNS])
        all_oos_predictions.append(test_df[["date", "ticker", "predicted_score", "actual_forward_return"]])

        final_model = model  # most recent fold's model is the deployable one

    oos_df = pd.concat(all_oos_predictions, ignore_index=True)
    OOS_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    oos_df.to_parquet(OOS_PREDICTIONS_PATH, index=False)
    print(f"[+] Saved {len(oos_df):,} out-of-sample predictions -> {OOS_PREDICTIONS_PATH}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"[+] Saved final model -> {MODEL_PATH}")
    print(f"[+] Model feature columns: {CS_FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
