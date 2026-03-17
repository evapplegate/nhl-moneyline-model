# NHL Moneyline Prediction Model

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)
[![Tests](https://github.com/evapplegate/nhl-moneyline-model/actions/workflows/tests.yml/badge.svg)](https://github.com/evapplegate/nhl-moneyline-model/actions)

A machine learning pipeline that predicts **NHL home-team win probabilities** using Elo ratings, rolling team form, and rest advantage. Trained on 6,557 games (Oct 2021 – Mar 2026) and deployed via FastAPI for real-time predictions.

---

## TL;DR: Key Results

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Log Loss** | 0.6799 | Model calibration (lower better) |
| **AUC** | 0.5819 | Discriminative power vs random (0.5) |
| **Brier Score** | 0.2435 | Prediction accuracy (lower better) |
| **Training Data** | 6,557 games | ~4.5 calendar years of NHL |
| **Teams** | 37 | Including 2025–26 expansion |

**Bottom Line:** Model achieves modest but real edge (~3% AUC improvement over random). NHL games are chaotic; full season simulation still beats sportsbook consensus.

---

## Architecture

```
Data Pipeline                      ML Pipeline                    API
──────────────────────────────────────────────────────────────────────────

NHL Public API                     Games CSVs                     Team State
(games, scores)   ──→  fetch_nhl    ──→  phase2_clean    ──→    (latest Elo,
                       _games_api       _and_clean           form, rest)
                                                               │
                                                               ↓
                            phase3_make_features         Team State + 
                       (Elo update, rolling         Upcoming Schedule
                        10-game windows)            │
                                │                   ↓
                                ↓               predict_
                        Games w/ Features      upcoming_
                            CSV                 games.py
                            │                   │
                            ↓                   ↓
                    phase4_train_model    Predictions CSV
                   (LogReg + XGBoost)    (281 games, 30d out)
                            │
                            ↓
                        models/
                   logreg_moneyline
                        .joblib         ──→ FastAPI Server
                                           (/predict endpoint)
                                           (real-time scoring)
```

---

## What This MVP Includes

### 1. **End-to-End Data Pipeline**
- Automated daily fetch from NHL's public schedule API
- Robust cleaning & type validation
- Feature engineering: Elo system, rolling stats, matchup deltas
- No data leakage (all features lagged appropriately)

### 2. **Two Trained Models**
- **Logistic Regression:** Calibrated, interpretable, production baseline
- **XGBoost:** Tree-based, non-linear, second opinion for ensemble

### 3. **Real-Time Predictions**
- REST API (FastAPI) scores live and upcoming games
- Team state updated post-game (Elo, form, rest)
- Response includes probability, features used, and edge analysis

### 4. **Evaluation Artifacts**
- ROC curves (both models)
- Calibration plot (predicted vs actual win rate)
- Feature importance (XGBoost)
- Confusion matrices & metrics table

### 5. **Documentation**
- This README (project overview)
- [MODEL_CARD.md](MODEL_CARD.md) (technical detailed spec)
- [notebooks/model-development.ipynb](notebooks/model-development.ipynb) (interactive walkthrough)

---

## Features & Rationale

### Core Features (from team state)
| Feature | Why | Range |
|---------|-----|-------|
| **Elo Differential** | Team strength signal; strong predictor | [-300, +300] |
| **Home/Away Rolling Win %** | Recent form > season average; 10-game window | [0%, 100%] |
| **Home/Away Rolling Goal Diff** | Expected goals proxy; power play indicator | [-2, +2] per game |
| **Rest Advantage** | Fatigue effects real; back-to-backs hurt | [-7, +7] days |

### Derived Features (learned by model)
- `form_diff`, `gd_diff`, `split_form_diff`, `split_gd_diff`
- Model learns which **matchup deltas** drive wins

### Why No Player Data?
MVP intentionally avoids:
- Injury reports (complex, real-time)
- Advanced stats (Corsi, xG, requires NHL.com scraping)
- Star player lineups (daily changes)

**Because:** Demonstrates clean ML pipeline on public data. Pro models layer these on top.

---

## Quickstart

### 1. Environment Setup
```bash
# Clone repo & navigate
cd ~/your-projects/nhl-moneyline-model

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
# Fetch (via NHL API), clean, engineer features, train
python src/run_pipeline.py

# Builds:
# - data/processed/games_clean.csv
# - data/processed/games_with_features.csv
# - models/logreg_moneyline.joblib
# - data/processed/team_state_latest.csv
```

### 3. Generate Evaluation Plots
```bash
python src/eval_models.py
# Outputs: reports/{roc_curves, calibration, confusion, importance}.png
```

### 4. Predict Upcoming Games
```bash
python src/predict_upcoming_games.py
# Default: predicts today's scheduled games (no team input required)
# Output: data/predictions_YYYY-MM-DD.csv

# Optional: predict a specific date
python src/predict_upcoming_games.py --date 2026-03-10

# Optional: predict a window of days (legacy upcoming behavior)
python src/predict_upcoming_games.py --date 2026-03-10 --days 30
# Output for multi-day runs: data/upcoming_predictions.csv

# Optional: pull sportsbook odds + edge + unit sizing (requires ODDS_API_KEY)
export ODDS_API_KEY="your_api_key_here"
python src/predict_upcoming_games.py --bookmaker draftkings --bankroll 1000 --unit-size-pct 0.01 --kelly-cap 0.25 --min-edge 0.02
```

When `ODDS_API_KEY` is set, prediction outputs include sportsbook pricing and recommendations:
- `home_odds_american`, `away_odds_american`
- `implied_home_prob`, `implied_away_prob`, `vig`
- `edge_home`, `edge_away`
- `recommended_stake`, `recommended_units`, `bet_side`, `recommend_bet`

### 5. Start API Server
```bash
uvicorn app.main:app --reload
# Open: http://127.0.0.1:8000/docs
```

### 6. Example API Calls

**Get available teams:**
```bash
curl http://127.0.0.1:8000/teams
# [{"teams": ["ANA", "ARI", "BOS", ..., "WPG"]}]
```

**Predict a game:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "game_date": "2026-03-10",
    "home_team": "BOS",
    "away_team": "NYR",
    "model": "logreg",
    "home_odds": -135,
    "away_odds": 115,
    "bankroll": 1000,
    "kelly_cap": 0.25
  }'
```

**Response includes:**
```json
{
  "home_team": "BOS",
  "away_team": "NYR",
  "p_home_win": 0.62,
  "p_away_win": 0.38,
  "features_used": {
    "elo_diff": 65,
    "home_rolling_win_pct": 0.6,
    ...
  },
  "edge_home": 0.05,
  "recommended_stake": 75.00,
  "recommend_bet": true
}
```

---

## Docker Deployment

### Run via Docker (Single Command)

No local Python setup required—Docker handles everything:

```bash
# Option 1: Using our convenience script
./docker-build-and-run.sh
# Opens http://localhost:8000/docs

# Option 2: Using docker-compose
docker-compose up --build
# Opens http://localhost:8000/docs

# Option 3: Manual build & run
docker build -t nhl-model .
docker run -p 8000:8000 nhl-model
```

**Why Docker?**
- Identical environment: laptop → cloud → interviewer's machine
- No dependency conflicts
- Easy deployment to AWS/GCP/Azure
- Clean isolation

---

## Running Tests

Automated test suite validates data quality, model integrity, and API functionality:

```bash
# Install test dependencies (included in requirements.txt)
pip install pytest httpx

# Run full test suite
pytest tests/ -v

# Run specific test module
pytest tests/test_data_pipeline.py -v
pytest tests/test_model_integrity.py -v
pytest tests/test_api.py -v
```

**Test Coverage (23 tests):**
- **Data Pipeline** (9 tests): Schema validation, null checks, feature integrity, home win baseline
- **Model Integrity** (8 tests): Model loading, predictions in [0,1], feature count, reproducibility
- **API Endpoints** (6 tests): Health check, teams endpoint (37+ teams), valid prediction schema, error handling, documentation

All tests pass locally: `23 passed`

---

## Continuous Integration & Deployment

GitHub Actions automatically tests on every push:

- **Tests Workflow** (`.github/workflows/tests.yml`):
  - Runs on: Python 3.11, Linux
  - Triggers: Every push to `main` or `develop`
  - Jobs: Install dependencies → Lint (flake8) → Run pytest (23 tests) → Upload coverage (`coverage.xml`)
  - Status: Check [Actions tab](https://github.com/evapplegate/nhl-moneyline-model/actions) for latest runs

- **Docker Build** (`.github/workflows/docker-build.yml`):
  - Manual trigger via "Actions" tab
  - Builds Docker image and verifies health endpoint
  - Output: Docker image ready for deployment

- **Weekly Retrain** (`.github/workflows/retrain.yml`):
  - Triggers: Weekly schedule (Mondays) + manual dispatch
  - Jobs: Run pipeline, regenerate metrics/artifacts, auto-commit if outputs change
  - Purpose: Keep model artifacts fresh with minimal maintenance overhead

---

## Interview Demo

For a concise interview walkthrough, command snippets, and expected API responses, see [DEMO.md](DEMO.md).

---

## Project Structure

```
.
├── README.md                      ← You are here
├── MODEL_CARD.md                  ← Detailed model spec
├── requirements.txt               ← Dependencies
├── LICENSE                        ← MIT
├── .gitignore
│
├── app/
│   └── main.py                    ← FastAPI server
│
├── src/
│   ├── fetch_nhl_games_api.py     ← Phase 1: Ingest (30 days of games)
│   ├── phase2_load_and_clean.py   ← Phase 2: Validation & type conversion
│   ├── phase3_make_features.py    ← Phase 3: Elo, rolling stats, rest
│   ├── phase4_train_model.py      ← Phase 4: LogReg + XGBoost training
│   ├── phase4_tune_elo.py         ← Hyperparameter grid search
│   ├── phase5_build_team_state.py ← Phase 5: Latest team snapshots
│   ├── run_pipeline.py            ← One-command runner (phases 1-5)
│   ├── eval_models.py             ← Generate plots & metrics
│   └── predict_upcoming_games.py  ← Score today's games (or any date/range)
│
├── notebooks/
│   └── model-development.ipynb    ← Interactive EDA + insights
│
├── data/
│   ├── raw/
│   │   └── nhl_games.csv          ← Fetched from API (~6500 rows)
│   ├── processed/
│   │   ├── games_clean.csv        ← Validated games
│   │   ├── games_with_features.csv ← Full feature matrix
│   │   ├── team_state_latest.csv  ← Current Elo + form per team
│   │   └── upcoming_predictions.csv ← Predictions for selected multi-day runs
│   └── raw/
│
├── models/
│   └── logreg_moneyline.joblib    ← Trained models (both algorithms)
│
└── reports/
    ├── roc_curves.png             ← Model comparison
    ├── calibration_plot.png       ← Predicted vs observed
    ├── confusion_matrices.png     ← TP/FP/TN/FN heatmaps
    ├── feature_importance.png     ← XGBoost Gini ranking
    └── metrics_comparison.csv     ← Side-by-side numbers
```

---

## Limitations & Future Work

### Current Limitations
- **Feature simplicity:** 14 engineered features (pro models use 100+)
- **No player data:** Can't account for injuries or star lineup changes
- **Modest AUC:** 0.58 is pedestrian; NHL inherently random
- **Weekly retraining only:** Doesn't react to mid-week roster moves

### High-Impact Improvements
- [ ] Integrate player-level corsi/xG data
- [ ] Goalie quality metric (GAA, sv% last 5 games)
- [ ] Injury impact modeling (absence of star players)
- [ ] Ensemble with betting market consensus (square vs sharp)
- [ ] Intra-season elo parameter auto-tuning

---

## Notes & Disclaimers

### What This Is
✓ Educational project demonstrating ML on real sports data  
✓ Portfolio piece showcasing data engineering + modeling  
✓ Functional prediction API for simulation/analysis  

### What This Is NOT
✗ Financial advice or gambling strategy  
✗ Sportsbook-beating algorithm (1-2% edge insufficient for vig)  
✗ Real-time injury/lineup reactive system  

### Ethical Use
- Use for **learning**, **simulation**, or **ensemble voting only**
- Do not trust single predictions for financial wagers
- Monitor for model drift and retrain if metrics decay

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11+ |
| **Data** | Pandas, NumPy |
| **ML** | scikit-learn, XGBoost |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Viz** | Matplotlib, Seaborn |
| **Dev** | Jupyter, Git |

---

## License

MIT License © 2026  
See [LICENSE](LICENSE) for details.

---

## Contact & Questions

For questions on:
- **Model architecture** → see [MODEL_CARD.md](MODEL_CARD.md)
- **Feature engineering** → see [notebooks/model-development.ipynb](notebooks/model-development.ipynb)
- **Data pipeline** → see source code comments in `src/`
- **Deployment** → run `uvicorn app.main:app --reload`

---

**Last Updated:** March 2, 2026  
**Maintained By:** [Your Name]  
**Status:** Production-Ready for Demo / Portfolio Use

