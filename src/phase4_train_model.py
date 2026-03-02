from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data/processed/games_with_features.csv"
MODEL_PATH = ROOT / "models/logreg_moneyline.joblib"

FEATURES = [
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
    "form_diff",
    "gd_diff",
    "split_form_diff",
    "split_gd_diff",
]

TARGET = "home_win"


def time_split(df: pd.DataFrame, test_size: float = 0.3):
    """
    Simple time-based split: earliest (1-test_size) for train, latest test_size for test.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run phase3_make_features.py first.")

    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Fill base features first (the original 9 that already exist in the CSV)
    base_features = [
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
    df[base_features] = df[base_features].fillna(0.0)

    # Now create diff features
    df["form_diff"] = df["home_rolling_win_pct"] - df["away_rolling_win_pct"]
    df["gd_diff"] = df["home_rolling_goal_diff"] - df["away_rolling_goal_diff"]
    df["split_form_diff"] = df["home_home_rolling_win_pct"] - df["away_away_rolling_win_pct"]
    df["split_gd_diff"] = df["home_home_rolling_goal_diff"] - df["away_away_rolling_goal_diff"]

    # Finally, if you want, fill any remaining NaNs in the full FEATURES list
    df[FEATURES] = df[FEATURES].fillna(0.0)


    train_df, test_df = time_split(df, test_size=0.3)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    # Pipeline: scale numeric features -> logistic regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    # Predict probabilities for home win
    p_test = model.predict_proba(X_test)[:, 1]

    # Metrics for probability models
    ll = log_loss(y_test, p_test)
    bs = brier_score_loss(y_test, p_test)
    auc = roc_auc_score(y_test, p_test) if len(set(y_test)) > 1 else float("nan")

    print("\n✅ Model trained: Logistic Regression")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print("\n📊 Evaluation on TEST set")
    print(f"Log loss:    {ll:.4f}  (lower is better)")
    print(f"Brier score: {bs:.4f}  (lower is better)")
    print(f"AUC:         {auc:.4f}  (higher is better)\n")

    # -----------------------------
    # XGBOOST MODEL (no scaling needed)
    # -----------------------------
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.5,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=42,
        eval_metric="logloss",
    )



    xgb.fit(X_train, y_train)






    p_test_xgb = xgb.predict_proba(X_test)[:, 1]

    ll_xgb = log_loss(y_test, p_test_xgb)
    bs_xgb = brier_score_loss(y_test, p_test_xgb)
    auc_xgb = roc_auc_score(y_test, p_test_xgb) if len(set(y_test)) > 1 else float("nan")

    print("\n🚀 XGBoost Evaluation on TEST set")
    print(f"Log loss:    {ll_xgb:.4f}  (lower is better)")
    print(f"Brier score: {bs_xgb:.4f}  (lower is better)")
    print(f"AUC:         {auc_xgb:.4f}  (higher is better)")

    out_xgb = test_df[["date", "home_team", "away_team", TARGET]].copy()
    out_xgb["p_home_win"] = p_test_xgb
    print("\n🔎 XGBoost sample predictions:")
    print(out_xgb.head(10).to_string(index=False))



    # Show a few example predictions
    out = test_df[["date", "home_team", "away_team", TARGET]].copy()
    out["p_home_win"] = p_test
    print("🔎 Sample predictions:")
    print(out.head(10).to_string(index=False))

    # Save the model artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"logreg": model, "xgb": xgb, "features": FEATURES},
        MODEL_PATH
    )   

    print(f"\n💾 Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
