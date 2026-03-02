"""
Predict outcomes for upcoming (unplayed) NHL games in the current season.
Loads team state and model, then fetches and scores scheduled games.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import math

import joblib
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/logreg_moneyline.joblib"
TEAM_STATE_PATH = ROOT / "data/processed/team_state_latest.csv"

HOME_ADV = 40.0  # Must match phase3_make_features.py

# Full feature set (matches what the model was trained on)
BASE_FEATURES = [
    "elo_diff",
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

FEATURES = BASE_FEATURES + [
    "form_diff",
    "gd_diff",
    "split_form_diff",
    "split_gd_diff",
]

BASE_URL = "https://api-web.nhle.com/v1/schedule"


def load_artifacts():
    """Load trained model and team state."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing {MODEL_PATH}. Run training pipeline first.")
    if not TEAM_STATE_PATH.exists():
        raise RuntimeError(f"Missing {TEAM_STATE_PATH}. Run phase5_build_team_state.py first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle.get("logreg") or bundle.get("xgb")
    if model is None:
        raise RuntimeError("No supported model found in artifact bundle.")

    state = pd.read_csv(TEAM_STATE_PATH)
    state["team"] = state["team"].astype(str)
    state["last_game_date"] = pd.to_datetime(state["last_game_date"], errors="coerce")
    state["current_elo"] = pd.to_numeric(state["current_elo"], errors="coerce").fillna(1500.0)

    return model, state.set_index("team")


def fetch_upcoming_games(start_date: date) -> list[dict]:
    """Fetch unplayed games from NHL API starting from start_date."""
    games = []
    
    # Fetch next 30 days of schedule
    current = start_date
    for _ in range(30):
        try:
            url = f"{BASE_URL}/{current.isoformat()}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Collect unplayed games (gameState != "OFF")
            for week in data.get("gameWeek", []):
                for game in week.get("games", []):
                    state = game.get("gameState", "").upper()
                    # Skip finished games
                    if state == "OFF":
                        continue
                    
                    start_time = game.get("startTimeUTC")
                    if not start_time:
                        continue
                    
                    game_date = start_time.split("T")[0]
                    home = game["homeTeam"]["abbrev"]
                    away = game["awayTeam"]["abbrev"]
                    
                    games.append({
                        "date": game_date,
                        "home_team": home,
                        "away_team": away,
                        "time_utc": start_time,
                    })
            
            current += timedelta(days=1)
        except Exception as e:
            print(f"⚠️ Failed to fetch {current}: {e}")
            current += timedelta(days=1)
            continue
    
    # Deduplicate by (date, home, away)
    seen = set()
    unique = []
    for g in games:
        key = (g["date"], g["home_team"], g["away_team"])
        if key not in seen:
            seen.add(key)
            unique.append(g)
    
    return unique


def predict_game(model, team_state: pd.DataFrame, game: dict) -> dict:
    """Predict home win probability for a single game."""
    home = game["home_team"].strip().upper()
    away = game["away_team"].strip().upper()
    game_dt = pd.to_datetime(game["date"])
    
    if home not in team_state.index or away not in team_state.index:
        return None  # Skip if teams unknown
    
    home_row = team_state.loc[home]
    away_row = team_state.loc[away]
    
    def rest_days(last_dt):
        if pd.isna(last_dt):
            return 3.0
        d = (game_dt - last_dt).days
        if d < 0:
            return 3.0
        return float(min(max(d, 0), 10))
    
    home_rest = rest_days(home_row.get("last_game_date"))
    away_rest = rest_days(away_row.get("last_game_date"))
    rest_diff = home_rest - away_rest
    
    home_elo = float(home_row.get("current_elo", 1500.0))
    away_elo = float(away_row.get("current_elo", 1500.0))
    elo_diff = (home_elo + HOME_ADV) - away_elo
    
    def getf(row, col, default):
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    
    # Get all features including home_home_* and away_away_*
    home_rolling_win = getf(home_row, "home_rolling_win_pct", 0.5)
    away_rolling_win = getf(away_row, "away_rolling_win_pct", 0.5)
    home_rolling_gd = getf(home_row, "home_rolling_goal_diff", 0.0)
    away_rolling_gd = getf(away_row, "away_rolling_goal_diff", 0.0)
    home_home_rolling_win = getf(home_row, "home_rolling_win_pct", 0.5)  # Use same as overall for single team
    home_home_rolling_gd = getf(home_row, "home_rolling_goal_diff", 0.0)
    away_away_rolling_win = getf(away_row, "away_rolling_win_pct", 0.5)
    away_away_rolling_gd = getf(away_row, "away_rolling_goal_diff", 0.0)
    
    x = {
        "elo_diff": elo_diff,
        "home_rolling_win_pct": home_rolling_win,
        "away_rolling_win_pct": away_rolling_win,
        "home_rolling_goal_diff": home_rolling_gd,
        "away_rolling_goal_diff": away_rolling_gd,
        "rest_diff": rest_diff,
        "home_home_rolling_win_pct": home_home_rolling_win,
        "home_home_rolling_goal_diff": home_home_rolling_gd,
        "away_away_rolling_win_pct": away_away_rolling_win,
        "away_away_rolling_goal_diff": away_away_rolling_gd,
    }
    
    # Compute diff features
    x["form_diff"] = x["home_rolling_win_pct"] - x["away_rolling_win_pct"]
    x["gd_diff"] = x["home_rolling_goal_diff"] - x["away_rolling_goal_diff"]
    x["split_form_diff"] = x["home_home_rolling_win_pct"] - x["away_away_rolling_win_pct"]
    x["split_gd_diff"] = x["home_home_rolling_goal_diff"] - x["away_away_rolling_goal_diff"]
    
    X = pd.DataFrame([x], columns=FEATURES)
    p_home = float(model.predict_proba(X)[:, 1][0])
    
    return {
        "date": game["date"],
        "home_team": home,
        "away_team": away,
        "time_utc": game["time_utc"],
        "p_home_win": p_home,
        "p_away_win": 1.0 - p_home,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "rest_diff": rest_diff,
    }


def main() -> None:
    print("🔮 Loading model and team state...")
    model, team_state = load_artifacts()
    
    today = date.today()
    print(f"📅 Fetching upcoming games from {today}...")
    games = fetch_upcoming_games(today)
    
    if not games:
        print("ℹ️ No upcoming games found.")
        return
    
    print(f"✅ Found {len(games)} upcoming games.\n")
    
    predictions = []
    for game in games:
        pred = predict_game(model, team_state, game)
        if pred is not None:
            predictions.append(pred)
    
    if not predictions:
        print("⚠️ Could not predict any games (teams unknown or other issue).")
        return
    
    df_pred = pd.DataFrame(predictions)
    df_pred = df_pred.sort_values("date").reset_index(drop=True)
    
    # Display summary
    print("="*100)
    print("UPCOMING GAMES - HOME WIN PREDICTIONS")
    print("="*100)
    
    # Group by date for readability
    for game_date in df_pred["date"].unique():
        print(f"\n📅 {game_date}")
        games_today = df_pred[df_pred["date"] == game_date]
        for _, row in games_today.iterrows():
            home_emoji = "✅" if row["p_home_win"] > 0.55 else "⚠️" if row["p_home_win"] > 0.45 else "❌"
            print(
                f"  {home_emoji} {row['home_team']:>3} vs {row['away_team']:<3} "
                f"| P(Home Win) = {row['p_home_win']:.1%} "
                f"| Elo: {row['home_elo']:.0f} vs {row['away_elo']:.0f} "
                f"| Rest Diff: {row['rest_diff']:+.0f}d"
            )
    
    print("\n" + "="*100)
    print(f"Total predictions: {len(df_pred)}")
    
    # Save to CSV
    output_path = ROOT / "data/upcoming_predictions.csv"
    df_pred.to_csv(output_path, index=False)
    print(f"💾 Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
