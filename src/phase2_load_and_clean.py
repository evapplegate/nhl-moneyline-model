from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/nhl_games.csv"
OUT_PATH = ROOT / "data/processed/games_clean.csv"

REQUIRED_COLS = {"date", "home_team", "away_team", "home_goals", "away_goals"}


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RAW_PATH}. Put a CSV there with columns: {sorted(REQUIRED_COLS)}"
        )

    df = pd.read_csv(RAW_PATH)

    # --- Basic validation ---
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # --- Parse types ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

    # Drop rows with bad data
    before = len(df)
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    after = len(df)

    # Remove rows where home_team == away_team (data errors)
    df = df[df["home_team"] != df["away_team"]].copy()

    # Create target: home win (ignore ties by dropping them)
    df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
    df = df[df["home_goals"] != df["away_goals"]].copy()

    # Keep only what we need for next steps
    df = df[["date", "home_team", "away_team", "home_goals", "away_goals", "home_win"]]
    df = df.sort_values("date").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✅ Loaded:", RAW_PATH)
    print(f"✅ Dropped {before - after} bad rows from parsing/NA")
    print("✅ Wrote:", OUT_PATH)
    print("\nPreview:")
    print(df.head(10))
    print("\nCounts:")
    print(df["home_win"].value_counts())


if __name__ == "__main__":
    main()
