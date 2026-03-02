from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data/processed/games_with_features.csv"
OUT_PATH = ROOT / "data/processed/team_state_latest.csv"

# These should match your model features present in games_with_features.csv
TEAM_FEATURES = [
    "rolling_win_pct",
    "rolling_goal_diff",
    "home_rolling_win_pct",
    "home_rolling_goal_diff",
    "away_rolling_win_pct",
    "away_rolling_goal_diff",
]

def main():
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # We need: last game date per team + latest features.
    # We'll build a "long" table: each row = team appearance in a game
    home = df[["date", "home_team"]].copy()
    home["team"] = home["home_team"]
    home = home[["date", "team"]]

    away = df[["date", "away_team"]].copy()
    away["team"] = away["away_team"]
    away = away[["date", "team"]]

    appearances = pd.concat([home, away], ignore_index=True)
    last_played = appearances.groupby("team")["date"].max().reset_index()
    last_played = last_played.rename(columns={"date": "last_game_date"})

    # We also want “latest row where team was home” and “latest row where team was away”
    # because you engineered home_* and away_* features.
    home_last = (
        df.sort_values("date")
          .groupby("home_team")
          .tail(1)[["home_team", "date", "home_rolling_win_pct", "home_rolling_goal_diff"]]
          .rename(columns={"home_team": "team", "date": "last_home_game_date"})
    )

    away_last = (
        df.sort_values("date")
          .groupby("away_team")
          .tail(1)[["away_team", "date", "away_rolling_win_pct", "away_rolling_goal_diff"]]
          .rename(columns={"away_team": "team", "date": "last_away_game_date"})
    )

    # Build current Elo per team using most recent post-game Elo
    home_last_elo = (
        df.sort_values("date")
        .groupby("home_team")
        .tail(1)[["home_team", "home_elo_post"]]
        .rename(columns={"home_team": "team", "home_elo_post": "home_last_elo"})
    )

    away_last_elo = (
        df.sort_values("date")
        .groupby("away_team")
        .tail(1)[["away_team", "away_elo_post"]]
        .rename(columns={"away_team": "team", "away_elo_post": "away_last_elo"})
    )

    elo_state = home_last_elo.merge(away_last_elo, on="team", how="outer")

    # If a team’s last game was as home, use home_last_elo; if away, use away_last_elo
    elo_state["current_elo"] = elo_state["home_last_elo"].combine_first(elo_state["away_last_elo"]).fillna(1500.0)


    # If you have per-game “elo_home” / “elo_away” columns in your file, use them.
    # If you only have elo_diff, we’ll approximate current Elo by carrying forward implied ratings is hard.
    # So: BEST is to store current team Elo in phase3; if you already do, rename below accordingly.
    elo_cols = [c for c in df.columns if c.lower() in ("home_elo", "away_elo", "elo_home", "elo_away")]
    if elo_cols:
        # Try to infer
        pass

    # Merge
    state = last_played.merge(home_last, on="team", how="left").merge(away_last, on="team", how="left")
    state = state.merge(elo_state[["team", "current_elo"]], on="team", how="left")
    state["current_elo"] = state["current_elo"].fillna(1500.0)


    # Fill missing values with neutral defaults
    state["home_rolling_win_pct"] = state["home_rolling_win_pct"].fillna(0.5)
    state["away_rolling_win_pct"] = state["away_rolling_win_pct"].fillna(0.5)
    state["home_rolling_goal_diff"] = state["home_rolling_goal_diff"].fillna(0.0)
    state["away_rolling_goal_diff"] = state["away_rolling_goal_diff"].fillna(0.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    state.to_csv(OUT_PATH, index=False)
    print(f"✅ Wrote team state: {OUT_PATH} | teams={len(state)}")
    print(state.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
