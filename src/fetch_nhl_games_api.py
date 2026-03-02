from __future__ import annotations
from datetime import date, datetime, timedelta
import argparse

import csv
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data/raw/nhl_games.csv"

# This endpoint is documented in the public NHL Stats API docs community reference.
# We'll query one date at a time to keep it simple and reliable.

BASE_URL = "https://api-web.nhle.com/v1/schedule"

def offset_to_minutes(offset_raw) -> int:
    """
    Convert various offset formats to minutes.
    Handles:
      - ints like -5 (hours)
      - ints like -300 (minutes)
      - strings like "-04:00" or "+01:30"
      - strings like "-5"
    """
    if offset_raw is None:
        return 0

    # If already numeric
    if isinstance(offset_raw, (int, float)):
        offset = float(offset_raw)
        # if looks like hours
        if abs(offset) <= 24:
            return int(offset * 60)
        return int(offset)

    s = str(offset_raw).strip()

    # Format like "-04:00" or "+01:30"
    if ":" in s:
        sign = -1 if s.startswith("-") else 1
        s2 = s.lstrip("+-")
        try:
            hh, mm = s2.split(":")
            return sign * (int(hh) * 60 + int(mm))
        except ValueError:
            return 0

    # Plain string number like "-5" or "-300"
    try:
        offset = int(s)
    except ValueError:
        return 0

    if abs(offset) <= 24:
        return offset * 60
    return offset


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def fetch_day(d: date) -> dict:
    url = f"{BASE_URL}/{d.isoformat()}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch completed NHL games from public API")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 10, 1), help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 4, 30), help="End date YYYY-MM-DD")
    return parser.parse_args()

def main():
    args = parse_args()
    start_date = args.start
    end_date = args.end

    if start_date > end_date:
        raise ValueError("--start must be <= --end")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    games_by_id = {}
    for d in daterange(start_date, end_date):
        try:
            data = fetch_day(d)
        except Exception as e:
            print(f"⚠️ Failed {d}: {e}")
            continue

        for block in data.get("gameWeek", []):
            for game in block.get("games", []):
                state = (game.get("gameState") or "").upper()
                if state != "OFF":
                    continue

                start_time = game.get("startTimeUTC")
                if not start_time:
                    continue

                # Parse UTC timestamp like "2023-10-10T21:30:00Z"
                dt_utc = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

                # venueUTCOffset sometimes arrives as a string (e.g., "-5")
                # Prefer venue offset; fallback to eastern offset if venue is missing
                offset_raw = game.get("venueUTCOffset", None)
                if offset_raw is None:
                    offset_raw = game.get("easternUTCOffset", 0)

                offset_minutes = offset_to_minutes(offset_raw)

                dt_local = dt_utc + timedelta(minutes=offset_minutes)
                local_date = dt_local.date().isoformat()


                # Keep only games that belong to the requested LOCAL date
                if local_date != d.isoformat():
                    continue


                home = game["homeTeam"]["abbrev"]
                away = game["awayTeam"]["abbrev"]
                home_goals = game["homeTeam"].get("score")
                away_goals = game["awayTeam"].get("score")

        # Defensive: skip if scores missing
                if home_goals is None or away_goals is None:
                    continue

                game_id = game.get("id")
                if game_id is None:
                    continue

                start_time = game.get("startTimeUTC")
                if not start_time:
                    continue

                game_date = start_time.split("T")[0]  # UTC date is fine for labeling

                home = game["homeTeam"]["abbrev"]
                away = game["awayTeam"]["abbrev"]
                home_goals = game["homeTeam"].get("score")
                away_goals = game["awayTeam"].get("score")

                if home_goals is None or away_goals is None:
                    continue

                # ✅ Deduplicate by unique game id
                games_by_id[game_id] = {
                    "date": game_date,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }

                                    


        # small sleep to be polite
        time.sleep(0.05)

        if d.day == 1:
            print(f"Pulled through {d.isoformat()} | rows so far: {len(games_by_id)}")

    # Write CSV in the exact schema your pipeline expects
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date","home_team","away_team","home_goals","away_goals"])
        writer.writeheader()
        rows = sorted(games_by_id.values(), key=lambda x: (x["date"], x["home_team"], x["away_team"]))
        writer.writerows(rows)

    print(f"\n✅ Wrote {len(rows)} games to {OUT_PATH}")

if __name__ == "__main__":
    main()
