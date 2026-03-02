from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data/processed/games_with_features.csv"  # must contain base features + date/team columns

# Use the same features you already use EXCEPT elo_diff (we will regenerate it)
BASE_FEATURES = [
    "home_rolling_win_pct",
    "away_rolling_win_pct",
    "home_rolling_goal_diff",
    "away_rolling_goal_diff",
    "rest_diff",
    "home_home_rolling_win_pct",
    "home_home_rolling_goal_diff",
    "away_away_rolling_win_pct",
    "away_away_rolling_goal_diff",
]
TARGET = "home_win"

ELO_START = 1500.0

def nhl_season_id(dt: pd.Timestamp) -> int:
    return dt.year if dt.month >= 7 else dt.year - 1

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def compute_elo_diff(df: pd.DataFrame, k: float, home_adv: float, season_reg: float) -> pd.Series:
    df["game_key"] = (
        df["date"].astype(str) + "|" +
        df["home_team"].astype(str) + "|" +
        df["away_team"].astype(str) + "|" +
        df["home_goals"].astype(str) + "|" +
        df["away_goals"].astype(str)
    )

    
    df = df.sort_values(["date", "game_key"], kind="mergesort").reset_index(drop=True)
    seasons = df["date"].apply(nhl_season_id)

    ratings: dict[str, float] = {}
    current_season = None
    out = []

    for i, row in enumerate(df.itertuples(index=False)):
        season = seasons.iat[i]

        if current_season is None:
            current_season = season
        elif season != current_season:
            for t in list(ratings.keys()):
                ratings[t] = ELO_START + (ratings[t] - ELO_START) * season_reg
            current_season = season

        home = row.home_team
        away = row.away_team
        y = int(row.home_win)

        r_home = ratings.get(home, ELO_START)
        r_away = ratings.get(away, ELO_START)

        out.append((r_home + home_adv) - r_away)

        p_home = expected_score(r_home + home_adv, r_away)
        ratings[home] = r_home + k * (y - p_home)
        ratings[away] = r_away + k * ((1 - y) - (1 - p_home))

    return pd.Series(out)

def time_split(df: pd.DataFrame, test_size: float = 0.3):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def eval_combo(df: pd.DataFrame, k: float, home_adv: float, season_reg: float):
    # recompute elo_diff for this combo (no leakage)
    df = df.sort_values(["date", "home_team", "away_team"], kind = "mergesort").reset_index(drop=True)
    df["elo_diff"] = compute_elo_diff(df, k=k, home_adv=home_adv, season_reg=season_reg)

    # fill missing base features just in case
    df[BASE_FEATURES] = df[BASE_FEATURES].fillna(0.0)

    features = ["elo_diff"] + BASE_FEATURES

    train_df, test_df = time_split(df, test_size=0.3)

    X_train, y_train = train_df[features], train_df[TARGET].astype(int)
    X_test, y_test = test_df[features], test_df[TARGET].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    ll = log_loss(y_test, p)
    bs = brier_score_loss(y_test, p)
    auc = roc_auc_score(y_test, p) if len(set(y_test)) > 1 else float("nan")
    return ll, bs, auc

def main():
    # Use games_with_features.csv because it already contains your engineered rolling/rest features.
    # It should still have date/home_team/away_team/home_win.
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run phase3_make_features.py first.")

    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    need = ["date", "home_team", "away_team", "home_win"] + BASE_FEATURES
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {IN_PATH}: {missing}")

    # Small but useful grid (fast). Expand later if you want.
    K_GRID = [10, 15, 20, 25, 30]
    HOME_ADV_GRID = [40, 55, 65, 75, 90]
    SEASON_REG_GRID = [0.60, 0.70, 0.75, 0.80, 0.90]

    results = []
    best = None

    for k in K_GRID:
        for ha in HOME_ADV_GRID:
            for sr in SEASON_REG_GRID:
                ll, bs, auc = eval_combo(df, k=k, home_adv=ha, season_reg=sr)
                results.append((ll, bs, auc, k, ha, sr))
                if best is None or ll < best[0]:
                    best = (ll, bs, auc, k, ha, sr)
                print(f"K={k:>2} HA={ha:>3} SR={sr:.2f} | logloss={ll:.4f} brier={bs:.4f} auc={auc:.4f}")

    results.sort(key=lambda x: x[0])
    print("\n🏆 Best by log loss:")
    ll, bs, auc, k, ha, sr = results[0]
    print(f"logloss={ll:.4f} brier={bs:.4f} auc={auc:.4f} | K={k} HOME_ADV={ha} SEASON_REGRESSION={sr}")

    print("\nTop 10 combos:")
    for r in results[:10]:
        ll, bs, auc, k, ha, sr = r
        print(f"logloss={ll:.4f} brier={bs:.4f} auc={auc:.4f} | K={k} HA={ha} SR={sr:.2f}")

if __name__ == "__main__":
    main()
