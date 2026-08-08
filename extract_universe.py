"""
One-time local script: extracts the trained universe's ticker list from
features_dataset.parquet into a small text file. The live daemon only ever
needed the ticker names, not the full 150MB+ feature dataset -- this lets
the repo stay small enough to push to GitHub without needing Git LFS.

Run this once locally. data/universe.txt is small and safe to commit;
features_dataset.parquet itself should stay out of git (see .gitignore).
"""

from pathlib import Path

import pandas as pd

FEATURES_PATH = Path("data") / "features_dataset.parquet"
UNIVERSE_PATH = Path("data") / "universe.txt"


def main():
    df = pd.read_parquet(FEATURES_PATH, columns=["ticker"])
    tickers = sorted(df["ticker"].unique().tolist())

    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_PATH.write_text("\n".join(tickers))

    print(f"[+] Wrote {len(tickers)} tickers to {UNIVERSE_PATH}")


if __name__ == "__main__":
    main()
