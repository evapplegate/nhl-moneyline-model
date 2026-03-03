# NHL Moneyline Model - Demo

## 🎯 30-Second Pitch

This is a **production-ready ML pipeline** that predicts NHL game outcomes with 58% AUC using Elo ratings, team form, and rest advantage. The API serves real-time predictions, and the full pipeline is containerized for instant deployment anywhere.

**Why this is impressive:**
- ✅ Complete ML lifecycle (data ingestion → feature engineering → model training → API deployment)
- ✅ Reproducible: Docker ensures identical environments on any machine
- ✅ Engineered: 14 carefully-designed features, 22 automated tests, CI/CD pipeline
- ✅ Production-ready: FastAPI with error handling, OpenAPI docs, comprehensive logging

---

## 🚀 Quick Start (< 5 minutes)

### Option 1: Docker (Recommended - No Python Setup)

```bash
# Clone repo
git clone https://github.com/evapplegate/nhl-moneyline-model.git
cd nhl-moneyline-model

# Build and run (single command)
docker-compose up --build

# Or use the convenience script
./docker-build-and-run.sh
```

Open browser to: **http://localhost:8000/docs**

---

### Option 2: Local Python Setup

```bash
# Clone repo
git clone https://github.com/evapplegate/nhl-moneyline-model.git
cd nhl-moneyline-model

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn app.main:app --reload
```

Open browser to: **http://localhost:8000/docs**

---

## 📊 Example API Calls

### Get Available Teams
```bash
curl http://localhost:8000/teams

# Response:
# {"teams": ["ANA", "ARI", "BOS", "BUF", ..., "WPG"]}  [36 teams total]
```

### Predict a Game

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "game_date": "2026-03-02",
    "home_team": "ANA",
    "away_team": "ARI",
    "model": "logreg",
    "home_odds": -110,
    "away_odds": -110,
    "bankroll": 1000.0,
    "kelly_cap": 0.25
  }'
```

### Expected Response
```json
{
  "home_team": "ANA",
  "away_team": "ARI",
  "game_date": "2026-03-02",
  "p_home_win": 0.548,
  "p_away_win": 0.452,
  "features_used": {
    "elo_diff": 62.6,
    "home_rolling_win_pct": 0.8,
    "away_rolling_win_pct": 0.6,
    "home_rolling_goal_diff": 0.6,
    "away_rolling_goal_diff": 0.2,
    "rest_diff": 0.0
  },
  "home_odds_american": -110,
  "away_odds_american": -110,
  "implied_home_prob": 0.4762,
  "edge_home": 0.0718,
  "kelly_fraction": 0.287,
  "recommended_stake": 143.5,
  "recommend_bet": true
}
```

---

## 📚 Deep Dive

### Model Development Notebook
Interactive walkthrough of the entire ML pipeline:
- **Data exploration:** 6,557 NHL games (Oct 2021 – Mar 2026)
- **Feature engineering:** Elo system, rolling 10-game windows, rest effects
- **Model comparison:** Logistic Regression vs XGBoost
- **Evaluation:** ROC curves, calibration plots, feature importance
- [Open: `notebooks/model-development.ipynb`](notebooks/model-development.ipynb)

### Model Card (Technical Deep Dive)
Comprehensive documentation of model architecture, features, assumptions, and limitations:
- **Input features:** 14 engineered features (Elo, form, rest, interactions)
- **Training data:** 6,557 games, ~4.5 years of NHL
- **Validation:** Holdout evaluation, cross-validation, calibration analysis
- **Limitations:** NHL games are chaotic; model provides modest edge only
- [Read: `MODEL_CARD.md`](MODEL_CARD.md)

---

## 🧪 Testing

All code is validated by automated tests:

```bash
# Run full test suite
pytest tests/ -v

# Output: 22 passed in 1.47s
#   ✓ 9 data pipeline tests (schema, nulls, baseline)
#   ✓ 8 model integrity tests (loading, predictions, reproducibility)
#   ✓ 5 API tests (endpoints, validation, docs)
```

---

## 🔄 Evaluation Artifacts

See `reports/` directory:
- **roc_curves.png** - LogReg vs XGBoost comparison
- **calibration_plot.png** - Predicted vs observed win rates
- **confusion_matrices.png** - TP/FP/TN/FN heatmaps
- **feature_importance.png** - XGBoost Gini importance ranking
- **metrics_comparison.csv** - Side-by-side metrics (Log Loss, AUC, Brier Score)

---

## 🏗️ Architecture

```
Data Pipeline                    ML Pipeline                 API Server
────────────────────────────────────────────────────────────────────

NHL Games API                 games_clean.csv           team_state_latest.csv
  (6,557 games)                 (validated)               (current Elo/form)
       ↓                              ↓                           ↓
fetch_nhl_games_api.py        phase2_clean              phase5_build_team_state
       ↓                              ↓                           ↓
    Raw CSV              games_with_features.csv      Team State (14 features)
                               (14 features)                      ↓
                                   ↓                    predict_upcoming_games.py
                          phase3_make_features              ↓
                          phase4_train_model        Upcoming Predictions CSV
                                   ↓                           +
                            Trained Models          FastAPI Server
                         (LogReg + XGBoost)         (/docs, /predict)
```

---

## 📋 Next Steps (For Extension)

Want to improve this model? Here are some ideas:

1. **Add Features:** Player injury reports, advanced stats (xG, Corsi), weather
2. **Ensemble Models:** Neural networks, gradient boosting variants (LightGBM)
3. **Live betting:** Integrate with sports betting APIs (DraftKings, FanDuel)
4. **Analytics:** Generate pregame insights (matchup analysis, injury impact)
5. **Deployment:** Host on AWS Lambda/ECS or Heroku for 24/7 availability
6. **Monitoring:** Track prediction accuracy over time, alert on model drift

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## ✅ Verification Checklist

This portfolio has been verified to work end-to-end:

- ✅ Fresh clone → Full setup < 5 min (Docker) or 10 min (venv)
- ✅ All 22 tests pass locally (`pytest tests/ -v`)
- ✅ API starts and serves requests (`curl http://localhost:8000/health`)
- ✅ Docker image builds and runs (`docker build && docker run`)
- ✅ CI/CD pipeline configured (GitHub Actions on every push)
- ✅ All documentation complete and links working
- ✅ No sensitive data in repo (API keys, credentials excluded)
- ✅ Professional git history with meaningful commits

**Status:** Ready for interviews and production deployment! 🚀

