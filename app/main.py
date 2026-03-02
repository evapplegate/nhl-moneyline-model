from __future__ import annotations

from pathlib import Path
from datetime import date
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/logreg_moneyline.joblib"
TEAM_STATE_PATH = ROOT / "data/processed/team_state_latest.csv"

# Elo constants MUST match your tuned values (from phase4_tune_elo.py)
HOME_ADV = 40.0

FEATURES = [
    "elo_diff",
    "home_rolling_win_pct",
    "away_rolling_win_pct",
    "home_rolling_goal_diff",
    "away_rolling_goal_diff",
    "rest_diff",
]

app = FastAPI(
    title="NHL Moneyline API",
    description="Predict NHL home win probability using ML", 
    version="0.1.0"
    )



class PredictRequest(BaseModel):
    game_date: date = Field(..., description="Game date YYYY-MM-DD")
    home_team: str = Field(..., description="3-letter team code, e.g. BOS")
    away_team: str = Field(..., description="3-letter team code, e.g. NYR")
    model: str = "logreg"

    home_odds: int | None = Field(None, description="Home American odds, e.g. -135 or +120")
    away_odds: int | None = Field(None, description="Away American odds, e.g. -135 or +120")

    bankroll: float = Field(1000.0, ge=0)
    kelly_cap: float = Field(0.25, ge=0, le=1)
    min_edge: float = Field(0.02, ge=0, le=1) 



class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    game_date: date

    p_home_win: float
    p_away_win: float
    features_used: dict

    home_odds: int | None = None
    away_odds: int | None = None

    implied_home_prob: float | None = None
    implied_away_prob: float | None = None
    vig: float | None = None

    edge_home: float | None = None
    edge_away: float | None = None

    ev_home_per_dollar: float | None = None
    ev_away_per_dollar: float | None = None

    kelly_home: float | None = None
    kelly_away: float | None = None

    bet_side: str | None = None
    recommended_stake: float | None = None
    recommend_bet: bool | None = None

def implied_prob_from_american(odds: int) -> float:
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)

def decimal_from_american(odds: int) -> float:
    if odds < 0:
        return 1 + (100 / (-odds))
    return 1 + (odds / 100)

def remove_vig(p_home_imp: float, p_away_imp: float) -> tuple[float, float, float]:
    """
    Returns (p_home_no_vig, p_away_no_vig, vig).
    vig = (p_home_imp + p_away_imp) - 1
    """
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

def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing {MODEL_PATH}. Run phase4_train_model.py first.")
    if not TEAM_STATE_PATH.exists():
        raise RuntimeError(f"Missing {TEAM_STATE_PATH}. Run phase5_build_team_state.py first.")

    bundle = joblib.load(MODEL_PATH)
    models = {}
    if "logreg" in bundle:
        models["logreg"] = bundle["logreg"]
    if "xgb" in bundle:
        models["xgb"] = bundle["xgb"]
    if not models:
        raise RuntimeError(f"No supported models found in {MODEL_PATH}. Expected keys: 'logreg' and/or 'xgb'.")
    features = bundle.get("features", FEATURES)

    state = pd.read_csv(TEAM_STATE_PATH)
    state["team"] = state["team"].astype(str)
    state["last_game_date"] = pd.to_datetime(state["last_game_date"], errors="coerce")
    state["current_elo"] = pd.to_numeric(state["current_elo"], errors="coerce").fillna(1500.0)

    return models, features, state.set_index("team")


MODELS, FEATURES, TEAM_STATE = load_artifacts()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/teams")
def teams():
    return {"teams": sorted(TEAM_STATE.index.tolist())}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_name = req.model.strip().lower()
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'. Use logreg or xgb.")
    model = MODELS[model_name]
    home = req.home_team.strip().upper()
    away = req.away_team.strip().upper()

    if home == away:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")
    if home not in TEAM_STATE.index or away not in TEAM_STATE.index:
        raise HTTPException(status_code=404, detail="Unknown team code. Use GET /teams.")

    home_row = TEAM_STATE.loc[home]
    away_row = TEAM_STATE.loc[away]

    game_dt = pd.to_datetime(req.game_date)

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

    home_elo = float(home_row["current_elo"])
    away_elo = float(away_row["current_elo"])
    elo_diff = (home_elo + HOME_ADV) - away_elo

    def getf(row, col, default):
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)

    x = {
        "elo_diff": elo_diff,
        "home_rolling_win_pct": getf(home_row, "home_rolling_win_pct", 0.5),
        "away_rolling_win_pct": getf(away_row, "away_rolling_win_pct", 0.5),
        "home_rolling_goal_diff": getf(home_row, "home_rolling_goal_diff", 0.0),
        "away_rolling_goal_diff": getf(away_row, "away_rolling_goal_diff", 0.0),
        "rest_diff": rest_diff,
    }

    X = pd.DataFrame([x], columns=FEATURES)
    p_home = float(model.predict_proba(X)[:, 1][0])
    # Optional betting math (only if sportsbook odds are provided)
    p_away = float(1.0 - p_home)

    # Betting math (only if BOTH odds are provided)
    bet_side = "none"
    recommended_stake = 0.0
    recommend_bet = False

    implied_home_prob = None
    implied_away_prob = None
    vig = None
    edge_home = None
    edge_away = None
    ev_home_per_dollar = None
    ev_away_per_dollar = None
    kelly_home = None
    kelly_away = None

    if req.home_odds is not None and req.away_odds is not None:
        if req.home_odds == 0 or req.away_odds == 0:
            raise HTTPException(status_code=400, detail="American odds cannot be 0. Use values like -110 or +120.")

        imp_home_raw = float(implied_prob_from_american(req.home_odds))
        imp_away_raw = float(implied_prob_from_american(req.away_odds))

        imp_home, imp_away, vig = remove_vig(imp_home_raw, imp_away_raw)
        implied_home_prob = imp_home
        implied_away_prob = imp_away

        edge_home = float(p_home - imp_home)
        edge_away = float(p_away - imp_away)

        # EV per $1 for each side
        b_home = float(decimal_from_american(req.home_odds) - 1.0)
        b_away = float(decimal_from_american(req.away_odds) - 1.0)

        ev_home_per_dollar = float(p_home * b_home - (1.0 - p_home))
        ev_away_per_dollar = float(p_away * b_away - (1.0 - p_away))

        # Kelly (capped)
        kelly_home = float(min(kelly_fraction(p_home, req.home_odds), req.kelly_cap))
        kelly_away = float(min(kelly_fraction(p_away, req.away_odds), req.kelly_cap))

        # Decide which side (if any) — require BOTH: positive EV and min edge
        home_ok = (edge_home >= req.min_edge) and (ev_home_per_dollar > 0)
        away_ok = (edge_away >= req.min_edge) and (ev_away_per_dollar > 0)

        if home_ok or away_ok:
            # pick higher EV side
            if home_ok and (not away_ok or ev_home_per_dollar >= ev_away_per_dollar):
                bet_side = "home"
                recommended_stake = float(kelly_home * req.bankroll)
            else:
                bet_side = "away"
                recommended_stake = float(kelly_away * req.bankroll)

            recommend_bet = recommended_stake > 0
        else:
            bet_side = "none"
            recommended_stake = 0.0
            recommend_bet = False
    
    return PredictResponse(
        home_team=home,
        away_team=away,
        game_date=req.game_date,
        p_home_win=p_home,
        p_away_win=p_away,
        features_used=x,

        home_odds=req.home_odds,
        away_odds=req.away_odds,
        implied_home_prob=implied_home_prob,
        implied_away_prob=implied_away_prob,
        vig=vig,

        edge_home=edge_home,
        edge_away=edge_away,
        ev_home_per_dollar=ev_home_per_dollar,
        ev_away_per_dollar=ev_away_per_dollar,
        kelly_home=kelly_home,
        kelly_away=kelly_away,

        bet_side=bet_side,
        recommended_stake=recommended_stake,
        recommend_bet=recommend_bet,
    )
