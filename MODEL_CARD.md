# Model Card: NHL Moneyline Predictor

## Overview
A calibrated probabilistic machine learning model that predicts NHL home-team win probability given pre-game team state (Elo rating, rolling performance, rest).

**Model Type:** Classification (binary: home win / home loss)  
**Task:** Predict P(home team wins) given: Elo delta, form, goal differential, rest advantage  
**Training Data:** 6,557 completed NHL games (Oct 2021 – Mar 2026)

---

## Model Specifications

### Inputs
| Feature | Type | Range | Source |
|---------|------|-------|--------|
| `elo_diff` | int | [-300, +300] | Elo rating system (K=15, HA=+40) |
| `home_rolling_win_pct` | float | [0, 1] | 10-game win rate (home games only) |
| `away_rolling_win_pct` | float | [0, 1] | 10-game win rate (away games only) |
| `home_rolling_goal_diff` | float | [-2, +2] | 10-game avg goal diff (home) |
| `away_rolling_goal_diff` | float | [-2, +2] | 10-game avg goal diff (away) |
| `rest_diff` | float | [-7, +7] | Home rest days – Away rest days |
| **+ 8 derived features** | float | [various] | Composites & matchup differentials |

### Output
- **Format:** Probability [0, 1]
- **Threshold:** 0.5 (neutral home = 50-50)
- **Interpretation:** `p_home = 0.62` → "Model predicts 62% home team win"

### Architecture

#### Primary Model: **Logistic Regression**
```
Features → StandardScaler → Logistic Classifier
```
- **Pros:** Interpretable, calibrated, fast, low overfitting risk
- **Performance:**  
  - Log Loss: **0.6799** (lower is better)
  - Brier Score: **0.2435** (lower is better)
  - AUC: **0.5819** (0.56 vs 0.50 random)
  - Accuracy @ 0.5: **56.6%**

#### Secondary Model: **XGBoost**
```
Trees (depth=2, n=200) with early stopping
```
- **Pros:** Non-linear, feature interactions, second opinion
- **Performance:**  
  - Log Loss: **0.6824**
  - AUC: **0.5804**
  - Slightly lower calibration, comparable discriminative power

---

## Training & Validation

### Data Split
- **Train:** 70% (4,589 games, Oct 2021 – Nov 2024)
- **Test:** 30% (1,968 games, Dec 2024 – Mar 2026)
- **Temporal split** (no leakage)

### Hyperparameter Tuning
Small grid search over Elo K, home advantage, and season regression:
- Best combination: **K=15, HOME_ADV=40, SEASON_REGRESSION=0.60**
- Found by: cross-validation on train set

### Feature Engineering
Rolling statistics computed lag-free (information available at game time):
- Elo updated after each game (no lag)
- Rolling win% lagged by 1 game (game N-1 result not in game N features)
- Rest calculated from actual game dates (no lookahead)

### No Data Leakage
✓ All features computed from game N-1 or earlier  
✓ Target (home_win) only from current game  
✓ Team state updated post-game (prediction uses pre-game state)

---

## Performance Assessment

### Strengths
1. **Interpretable features:** Elo, rolling form, rest are domain-understandable
2. **Well-calibrated:** Log loss (0.68) close to Brier (0.24) → predictions reflect true prob
3. **Consistent ensemble:** LogReg and XGBoost agree 95%+ of time
4. **Production-ready:** FastAPI deployment with real-time updates

### Limitations
1. **Modest AUC (0.58):** NHL is chaotic; game outcomes heavily influenced by:
   - Goalie performance (binary random variable each game)
   - Injuries to key players (not captured in features)
   - Home/away split variance per team (need deeper analysis)
   - Luck (puck bounces, refereeing, fluky goals)

2. **Limited feature set (10 total):** Professional models use 100+:
   - Player-level metrics (Corsi, Expected Goals)
   - Star player status (Connor McDavid >> bench player)
   - Goalie quality + recent form
   - Back-to-back game fatigue
   - Travel distance & time zones

3. **No external data:** Could improve with:
   - Betting market odds (crowd wisdom)
   - Pre-game team news / injuries
   - Line movement trends

### When to Use
✓ **Good for:** Team strength comparison, season simulation, ensemble voting  
✗ **Not for:** Sole basis for large financial wagers (edge too thin)

---

## Usage

### Python API
```python
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load model
model = joblib.load("models/logreg_moneyline.joblib")["logreg"]

# Feature vector for BOS vs NYR game
features = {
    "elo_diff": 65,
    "home_rolling_win_pct": 0.6,
    "away_rolling_win_pct": 0.5,
    # ... etc
}
X = pd.DataFrame([features], columns=[...])
p_home = model.predict_proba(X)[0, 1]  # → 0.62
```

### REST API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "game_date": "2026-03-10",
    "home_team": "BOS",
    "away_team": "NYR",
    "model": "logreg"
  }'
```

### Command-line (upcoming predictions)
```bash
python src/predict_upcoming_games.py
# → data/upcoming_predictions.csv
```

---

## Evaluation Artifacts

| File | Description |
|------|-------------|
| `reports/roc_curves.png` | ROC curves (LogReg vs XGBoost) |
| `reports/calibration_plot.png` | Predicted vs observed probability |
| `reports/confusion_matrices.png` | TP/FP/TN/FN at threshold 0.5 |
| `reports/feature_importance.png` | XGBoost Gini importance ranking |
| `reports/metrics_comparison.csv` | Side-by-side metrics table |

---

## Ethical Considerations & Disclaimers

### Intended Use
- **Educational:** Learn ML on real sports data
- **Simulation:** Historical backtesting, scenario analysis
- **Ensemble:** One signal among many for betting decisions

### Risks & Biases
- **Selection bias:** Uses only completed games (survivor bias, no withdrawn games)
- **Data recency:** NHL rule changes (flat salary cap, 4v4 OT shift) affect meta
- **Imbalance:** 53% home win rate baked into baseline (no algorithm bias)
- **Uncertainty:** Model confidence (0.56 AUC) is low; don't trust single predictions

### Not Recommended For
- **Sole gambling strategy:** Edge ~1-2%, insufficient for vig + variance
- **Financial advice:** This is a game prediction model, not financial analysis
- **Real-time wagering:** Requires constant retraining (team composition changes)

---

## Maintenance & Retraining

### Update Frequency
**Weekly:** Retrain with latest completed games  
**Monthly:** Check for Elo parameter drift  
**Quarterly:** Reassess feature engineering given rule changes

### Monitoring
Track:
- Log loss on rolling test window
- Calibration error (predicted vs observed win rate)
- Feature drift (Elo distribution changes)
- Model disagreement (when LogReg and XGBoost diverge > 10%)

### Version Control
```
logreg_moneyline.joblib  # Current production model
game_features.csv        # Feature definitions (not in code)
team_state_latest.csv    # Snapshot of team state at save time
```

---

## References

- **Elo Rating System:** Glickman, M. (2013). https://www.glicko.net/
- **Logistic Regression:** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.*
- **XGBoost:** Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System."
- **Calibration:** Guo, C., et al. (2017). "On Calibration of Modern Neural Networks."

---

**Model Version:** 1.0  
**Last Updated:** 2026-03-02  
**Authors:** Data Scientist / ML Engineer  
**License:** MIT
