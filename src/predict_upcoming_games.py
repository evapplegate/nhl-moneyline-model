"""
Predict outcomes for upcoming (unplayed) NHL games in the current season.
Loads team state and model, then fetches and scores scheduled games.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import argparse
from pathlib import Path
import math
import os
from typing import Any

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
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"


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


def fetch_upcoming_games(start_date: date, days: int = 1) -> list[dict]:
    """Fetch unplayed games from NHL API starting from start_date for N days."""
    games = []
    end_date = start_date + timedelta(days=days - 1)

    # Fetch schedule window
    current = start_date
    for _ in range(days):
        try:
            url = f"{BASE_URL}/{current.isoformat()}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Collect unplayed games (gameState != "OFF")
            for week in data.get("gameWeek", []):
                week_date = week.get("date")
                if not week_date:
                    continue
                schedule_day = date.fromisoformat(week_date)
                if schedule_day < start_date or schedule_day > end_date:
                    continue

                for game in week.get("games", []):
                    state = game.get("gameState", "").upper()
                    # Skip finished games
                    if state == "OFF":
                        continue
                    
                    start_time = game.get("startTimeUTC")
                    if not start_time:
                        continue
                    
                    game_date = week_date
                    home = game["homeTeam"]["abbrev"]
                    away = game["awayTeam"]["abbrev"]
                    home_name = (
                        f"{game['homeTeam']['placeName']['default']} "
                        f"{game['homeTeam']['commonName']['default']}"
                    )
                    away_name = (
                        f"{game['awayTeam']['placeName']['default']} "
                        f"{game['awayTeam']['commonName']['default']}"
                    )
                    
                    games.append({
                        "date": game_date,
                        "home_team": home,
                        "away_team": away,
                        "home_name": home_name,
                        "away_name": away_name,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NHL games from the schedule without manually entering teams."
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date.today(),
        help="Start date in YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to predict starting at --date (default: 1).",
    )
    parser.add_argument(
        "--bookmaker",
        type=str,
        default="draftkings",
        help="Bookmaker key for odds provider (default: draftkings).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us",
        help="Odds region code (default: us).",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Bankroll used for stake sizing (default: 1000).",
    )
    parser.add_argument(
        "--unit-size-pct",
        type=float,
        default=0.01,
        help="1 unit as a fraction of bankroll (default: 0.01 = 1%%).",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.25,
        help="Maximum Kelly fraction cap (default: 0.25).",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.02,
        help="Minimum edge required to recommend a bet (default: 0.02).",
    )
    return parser.parse_args()


def implied_prob_from_american(odds: int) -> float:
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def decimal_from_american(odds: int) -> float:
    if odds < 0:
        return 1 + (100 / (-odds))
    return 1 + (odds / 100)


def remove_vig(p_home_imp: float, p_away_imp: float) -> tuple[float, float, float]:
    s = p_home_imp + p_away_imp
    if s <= 0:
        return 0.5, 0.5, 0.0
    p_home_nv = p_home_imp / s
    p_away_nv = p_away_imp / s
    vig = s - 1.0
    return p_home_nv, p_away_nv, vig


def kelly_fraction(p: float, american_odds: int) -> float:
    dec = decimal_from_american(american_odds)
    b = dec - 1.0
    if b <= 0:
        return 0.0
    k = (p * b - (1.0 - p)) / b
    return float(max(0.0, k))


def fetch_sportsbook_odds(
    start_date: date,
    end_date: date,
    bookmaker: str,
    region: str,
) -> dict[tuple[str, str, str], dict[str, int]]:
    """Fetch h2h odds from The Odds API and return map keyed by (date, home_name, away_name)."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ℹ️ ODDS_API_KEY not set. Skipping sportsbook odds and edge sizing.")
        return {}

    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": bookmaker,
        "dateFormat": "iso",
    }

    try:
        response = requests.get(ODDS_API_URL, params=params, timeout=20)
        response.raise_for_status()
        events: list[dict[str, Any]] = response.json()
    except Exception as exc:
        print(f"⚠️ Could not fetch sportsbook odds: {exc}")
        return {}

    odds_map: dict[tuple[str, str, str], dict[str, int]] = {}

    for event in events:
        commence = event.get("commence_time")
        if not commence:
            continue
        event_day = commence.split("T")[0]
        event_date = date.fromisoformat(event_day)
        if event_date < start_date or event_date > end_date:
            continue

        home_name = event.get("home_team")
        away_name = event.get("away_team")
        if not home_name or not away_name:
            continue

        home_odds = None
        away_odds = None
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team_name = outcome.get("name")
                    price = outcome.get("price")
                    if team_name == home_name:
                        home_odds = int(price)
                    elif team_name == away_name:
                        away_odds = int(price)

        if home_odds is None or away_odds is None:
            continue

        odds_map[(event_day, home_name, away_name)] = {
            "home_odds": home_odds,
            "away_odds": away_odds,
        }

    print(f"✅ Pulled sportsbook odds for {len(odds_map)} games from {bookmaker}.")
    return odds_map


def attach_betting_recommendation(
    prediction: dict,
    odds_info: dict[str, int],
    bankroll: float,
    unit_size_pct: float,
    kelly_cap: float,
    min_edge: float,
) -> dict:
    """Add edge and unit recommendation fields based on sportsbook odds."""
    p_home = float(prediction["p_home_win"])
    p_away = float(prediction["p_away_win"])
    home_odds = int(odds_info["home_odds"])
    away_odds = int(odds_info["away_odds"])

    imp_home_raw = implied_prob_from_american(home_odds)
    imp_away_raw = implied_prob_from_american(away_odds)
    imp_home, imp_away, vig = remove_vig(imp_home_raw, imp_away_raw)

    edge_home = p_home - imp_home
    edge_away = p_away - imp_away

    ev_home = p_home * (decimal_from_american(home_odds) - 1.0) - (1.0 - p_home)
    ev_away = p_away * (decimal_from_american(away_odds) - 1.0) - (1.0 - p_away)

    kelly_home = min(kelly_fraction(p_home, home_odds), kelly_cap)
    kelly_away = min(kelly_fraction(p_away, away_odds), kelly_cap)

    bet_side = "none"
    recommended_units = 0.0
    recommended_stake = 0.0

    home_ok = edge_home >= min_edge and ev_home > 0
    away_ok = edge_away >= min_edge and ev_away > 0

    unit_dollars = max(bankroll * unit_size_pct, 1e-9)
    if home_ok or away_ok:
        if home_ok and (not away_ok or ev_home >= ev_away):
            bet_side = "home"
            recommended_stake = bankroll * kelly_home
        else:
            bet_side = "away"
            recommended_stake = bankroll * kelly_away
        recommended_units = recommended_stake / unit_dollars

    prediction.update(
        {
            "home_odds_american": home_odds,
            "away_odds_american": away_odds,
            "implied_home_prob": imp_home,
            "implied_away_prob": imp_away,
            "vig": vig,
            "edge_home": edge_home,
            "edge_away": edge_away,
            "ev_home_per_dollar": ev_home,
            "ev_away_per_dollar": ev_away,
            "kelly_home": kelly_home,
            "kelly_away": kelly_away,
            "bet_side": bet_side,
            "recommended_stake": recommended_stake,
            "recommended_units": recommended_units,
            "recommend_bet": bet_side != "none",
        }
    )
    return prediction


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
    args = parse_args()
    if args.days < 1:
        raise ValueError("--days must be >= 1")
    if args.bankroll <= 0:
        raise ValueError("--bankroll must be > 0")
    if not (0 < args.unit_size_pct <= 1):
        raise ValueError("--unit-size-pct must be in (0, 1]")
    if not (0 <= args.kelly_cap <= 1):
        raise ValueError("--kelly-cap must be in [0, 1]")
    if not (0 <= args.min_edge <= 1):
        raise ValueError("--min-edge must be in [0, 1]")

    print("🔮 Loading model and team state...")
    model, team_state = load_artifacts()

    start_date = args.date
    end_date = start_date + timedelta(days=args.days - 1)
    print(f"📅 Fetching upcoming games from {start_date} to {end_date}...")
    games = fetch_upcoming_games(start_date, days=args.days)
    odds_map = fetch_sportsbook_odds(
        start_date=start_date,
        end_date=end_date,
        bookmaker=args.bookmaker,
        region=args.region,
    )
    
    if not games:
        print("ℹ️ No upcoming games found.")
        return
    
    print(f"✅ Found {len(games)} upcoming games.\n")
    
    predictions = []
    for game in games:
        pred = predict_game(model, team_state, game)
        if pred is not None:
            odds_info = odds_map.get((game["date"], game["home_name"], game["away_name"]))
            if odds_info is not None:
                pred = attach_betting_recommendation(
                    prediction=pred,
                    odds_info=odds_info,
                    bankroll=args.bankroll,
                    unit_size_pct=args.unit_size_pct,
                    kelly_cap=args.kelly_cap,
                    min_edge=args.min_edge,
                )
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
            edge_txt = ""
            if "edge_home" in row and pd.notna(row.get("edge_home")):
                side = str(row.get("bet_side", "none")).upper()
                units = float(row.get("recommended_units", 0.0))
                edge_txt = f" | Edge(H): {row['edge_home']:+.2%} | Bet: {side} {units:.2f}u"
            print(
                f"  {home_emoji} {row['home_team']:>3} vs {row['away_team']:<3} "
                f"| P(Home Win) = {row['p_home_win']:.1%} "
                f"| Elo: {row['home_elo']:.0f} vs {row['away_elo']:.0f} "
                f"| Rest Diff: {row['rest_diff']:+.0f}d"
                f"{edge_txt}"
            )
    
    print("\n" + "="*100)
    print(f"Total predictions: {len(df_pred)}")
    
    # Save to CSV
    output_name = (
        f"predictions_{start_date.isoformat()}.csv"
        if args.days == 1
        else "upcoming_predictions.csv"
    )
    output_path = ROOT / "data" / output_name
    df_pred.to_csv(output_path, index=False)
    print(f"💾 Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
