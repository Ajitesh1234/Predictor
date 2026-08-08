import pandas as pd
df = pd.read_parquet("data/features_dataset.parquet")
cols = ["close","actual_forward_return"] + [c+"_cs" for c in
    ["mom_12_1","mom_6_1","mom_1m_reversal","vol_60d","vol_120d",
     "rolling_beta","dist_52w_high","rsi_14","ma_50_200_cross"]]
for c in cols:
    if c in df.columns:
        print(f"{c}: {df[c].isna().mean():.1%} NaN")
    else:
        print(f"{c}: MISSING COLUMN")
