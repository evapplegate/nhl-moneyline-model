import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data/processed/games_clean.csv"
OUT_PATH = ROOT / "data/processed/games_with_features.csv"

ROLLING_WINDOW = 3  # small for demo


ELO_START = 1500.0
K = 15.0  # Tuned grid search (K=15 was optimal)
HOME_ADV = 40.0  # Tuned grid search (HOME_ADV=40 was optimal)
SEASON_REGRESSION = 0.60  # Tuned grid search (SR=0.60 was optimal)

def nhl_season_id(dt: pd.Timestamp) -> int:
    """
    NHL season spans roughly Oct->Jun.
    We'll label season by the starting year:
      Oct-Dec 2023 => season 2023
      Jan-Jun 2024 => season 2023
    """
    return dt.year if dt.month >= 7 else dt.year - 1

def expected_score(r_a: float, r_b: float) -> float:
    """Expected probability that A beats B given Elo ratings."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))



def main():
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["game_key"] = (
        df["date"].astype(str) + "|" +
        df["home_team"].astype(str) + "|" +
        df["away_team"].astype(str) + "|" +
        df["home_goals"].astype(str) + "|" +
        df["away_goals"].astype(str)
    )

    df = df.sort_values(["date", "game_key"], kind="mergesort").reset_index(drop=True)

    # -----------------------------
    # ELO FEATURES (NO LEAKAGE)
    # -----------------------------
    df["season"] = df["date"].apply(nhl_season_id)

    ratings: dict[str, float] = {}
    current_season = None

    home_elo_pre = []
    away_elo_pre = []
    home_elo_post = []
    away_elo_post = []
    elo_diff = []
    home_elo_list = []
    away_elo_list = []

    for row in df.itertuples(index=False):
        season = row.season

        # season change -> regress ratings toward mean
        if current_season is None:
            current_season = season
        elif season != current_season:
            for t in list(ratings.keys()):
                ratings[t] = ELO_START + (ratings[t] - ELO_START) * SEASON_REGRESSION
            current_season = season

        home = row.home_team
        away = row.away_team
        y = int(row.home_win)

        r_home = ratings.get(home, ELO_START)
        r_away = ratings.get(away, ELO_START)

        # store pre-game Elo features
        home_elo_pre.append(r_home)
        away_elo_pre.append(r_away)
        elo_diff.append((r_home + HOME_ADV) - r_away)
        home_elo_list.append(r_home)
        away_elo_list.append(r_away)

        # expected home win probability from Elo
        p_home = expected_score(r_home + HOME_ADV, r_away)

        # update Elo ratings AFTER using them (key for no leakage)
        r_home_new = r_home + K * (y - p_home)
        r_away_new = r_away + K * ((1 - y) - (1 - p_home))

        ratings[home] = r_home_new
        ratings[away] = r_away_new

        home_elo_post.append(ratings[home])
        away_elo_post.append(ratings[away])

    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    df["elo_diff"] = elo_diff
    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["home_elo_post"] = home_elo_post
    df["away_elo_post"] = away_elo_post


    # Expand dataset so each team appears once per game
    home_df = df[["date", "home_team", "away_team", "home_goals", "away_goals", "home_win"]].copy()
    home_df.columns = ["date", "team", "opponent", "goals_for", "goals_against", "win"]

    away_df = df[["date", "away_team", "home_team", "away_goals", "home_goals", "home_win"]].copy()
    away_df.columns = ["date", "team", "opponent", "goals_for", "goals_against", "home_win"]
    away_df["win"] = 1 - away_df["home_win"]
    away_df = away_df.drop(columns=["home_win"])

    home_df["is_home"] = 1
    away_df["is_home"] = 0

    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"])

    # Convert date to datetime
    long_df["date"] = pd.to_datetime(long_df["date"])

    # Compute days since last game per team
    long_df["rest_days"] = (
        long_df.groupby("team")["date"]
        .diff()
        .dt.days
    )

    # Fill first game rest with median rest (e.g. 3 days)
    long_df["rest_days"] = long_df["rest_days"].fillna(3)

    # Cap extreme rest (e.g., All-Star break / long gaps)
    long_df["rest_days_capped"] = long_df["rest_days"].clip(lower=0, upper=7)

    # Buckets: 0,1,2,3,4,5+
    long_df["rest_bucket"] = pd.cut(
        long_df["rest_days_capped"],
        bins=[-0.1, 0.5, 1.5, 2.5, 3.5, 4.5, 7.5],
        labels=["0", "1", "2", "3", "4", "5+"],
)





    WINDOW = 10

    long_df["rolling_win_pct"] = (
        long_df.groupby("team")["win"]
        .rolling(WINDOW, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]



    long_df["rolling_goal_diff"] = (
        long_df.groupby("team")["goal_diff"]
        .rolling(WINDOW, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # --- Split form features (home-only / away-only) ---
    home_only = long_df[long_df["is_home"] == 1].copy()
    home_only = home_only.sort_values(["team", "date"])

    home_only["home_only_rolling_win_pct"] = (
        home_only.groupby("team")["win"]
        .rolling(WINDOW, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    home_only["home_only_rolling_goal_diff"] = (
        home_only.groupby("team")["goal_diff"]
        .rolling(WINDOW, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    home_only_features = (
        home_only[["date", "team", "home_only_rolling_win_pct", "home_only_rolling_goal_diff"]]
        .drop_duplicates(subset=["date", "team"])
    )

    away_only = long_df[long_df["is_home"] == 0].copy()
    away_only = away_only.sort_values(["team", "date"])

    away_only["away_only_rolling_win_pct"] = (
        away_only.groupby("team")["win"]
        .rolling(WINDOW, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    away_only["away_only_rolling_goal_diff"] = (
        away_only.groupby("team")["goal_diff"]
        .rolling(WINDOW, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    away_only_features = (
        away_only[["date", "team", "away_only_rolling_win_pct", "away_only_rolling_goal_diff"]]
        .drop_duplicates(subset=["date", "team"])
    )



    # Merge back to original structure
    features = (
        long_df[["date", "team", "rolling_win_pct", "rolling_goal_diff", "rest_days", "rest_days_capped"]]
        .drop_duplicates(subset=["date", "team"])
    )

    home_features = features.rename(columns={
        "team": "home_team",
        "rolling_win_pct": "home_rolling_win_pct",
        "rolling_goal_diff": "home_rolling_goal_diff",
        "rest_days_capped": "home_rest_days"
    })

    away_features = features.rename(columns={
        "team": "away_team",
        "rolling_win_pct": "away_rolling_win_pct",
        "rolling_goal_diff": "away_rolling_goal_diff",
        "rest_days_capped": "away_rest_days"
    })

    df = df.merge(home_features, on=["date", "home_team"], how="left")
    df = df.merge(away_features, on=["date", "away_team"], how="left")
    df = df.drop(columns=["rest_days_y"], errors="ignore")

    # Merge home-team HOME-only form
    df = df.merge(
        home_only_features.rename(columns={
            "team": "home_team",
            "home_only_rolling_win_pct": "home_home_rolling_win_pct",
            "home_only_rolling_goal_diff": "home_home_rolling_goal_diff",
        }),
        on=["date", "home_team"],
        how="left",
    )

    # Merge away-team AWAY-only form
    df = df.merge(
        away_only_features.rename(columns={
            "team": "away_team",
            "away_only_rolling_win_pct": "away_away_rolling_win_pct",
            "away_only_rolling_goal_diff": "away_away_rolling_goal_diff",
        }),
        on=["date", "away_team"],
        how="left",
    )

    # Fill NaNs for early season / first games
    for col in [
        "home_home_rolling_win_pct",
        "home_home_rolling_goal_diff",
        "away_away_rolling_win_pct",
        "away_away_rolling_goal_diff",
    ]:
        df[col] = df[col].fillna(0.0)


    df["home_rest_days"] = df["home_rest_days"].fillna(3)
    df["away_rest_days"] = df["away_rest_days"].fillna(3)



    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]


    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✅ Features created.")
    print(df.head())


if __name__ == "__main__":
    main()
