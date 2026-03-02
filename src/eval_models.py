"""
Generate evaluation plots for trained models.
Saves visualizations to reports/ folder.

Generates:
  - ROC curves (LogRegression vs XGBoost)
  - Calibration plot (predicted vs observed)
  - Feature importance (XGBoost)
  - Confusion matrices
  - Model comparison metrics table
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss, log_loss, roc_auc_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/logreg_moneyline.joblib"
DATA_PATH = ROOT / "data/processed/games_with_features.csv"
REPORTS_PATH = ROOT / "reports"

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
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main() -> None:
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load data and models
    print("📊 Loading data and models...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    
    bundle = joblib.load(MODEL_PATH)
    logreg = bundle["logreg"]
    xgb = bundle["xgb"]
    
    # Fill base features first
    base_features = [f for f in FEATURES if f in df.columns]
    df[base_features] = df[base_features].fillna(0.0)
    
    # Create diff features matching phase4_train_model.py
    df["form_diff"] = df["home_rolling_win_pct"] - df["away_rolling_win_pct"]
    df["gd_diff"] = df["home_rolling_goal_diff"] - df["away_rolling_goal_diff"]
    df["split_form_diff"] = df["home_home_rolling_win_pct"] - df["away_away_rolling_win_pct"]
    df["split_gd_diff"] = df["home_home_rolling_goal_diff"] - df["away_away_rolling_goal_diff"]
    
    # Final fill for all features
    df[FEATURES] = df[FEATURES].fillna(0.0)
    
    # Time split (same as training)
    train_df, test_df = time_split(df, test_size=0.3)
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)
    
    # Get predictions
    p_logreg = logreg.predict_proba(X_test)[:, 1]
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    
    # ========== ROC CURVES ==========
    print("🎯 Generating ROC curves...")
    fpr_lr, tpr_lr, _ = roc_curve(y_test, p_logreg)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, p_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_lr:.3f})", linewidth=2)
    ax.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_auc_xgb:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label="Random Classifier", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves: Home Win Probability Prediction", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "roc_curves.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {REPORTS_PATH / 'roc_curves.png'}")
    plt.close()
    
    # ========== CALIBRATION PLOTS ==========
    print("📈 Generating calibration plot...")
    cal_lr_edges, cal_lr_true = calibration_curve(y_test, p_logreg, n_bins=10, strategy='quantile')
    cal_xgb_edges, cal_xgb_true = calibration_curve(y_test, p_xgb, n_bins=10, strategy='quantile')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration", linewidth=1.5)
    ax.plot(cal_lr_edges, cal_lr_true, 'o-', label="Logistic Regression", linewidth=2, markersize=6)
    ax.plot(cal_xgb_edges, cal_xgb_true, 's-', label="XGBoost", linewidth=2, markersize=6)
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Frequency (Actual Win Rate)", fontsize=11)
    ax.set_title("Calibration Plot: Prediction vs Reality", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "calibration_plot.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {REPORTS_PATH / 'calibration_plot.png'}")
    plt.close()
    
    # ========== CONFUSION MATRICES ==========
    print("🔍 Generating confusion matrices...")
    pred_logreg = (p_logreg >= 0.5).astype(int)
    pred_xgb = (p_xgb >= 0.5).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    cm_lr = confusion_matrix(y_test, pred_logreg)
    ConfusionMatrixDisplay(cm_lr, display_labels=["Home Loss", "Home Win"]).plot(ax=axes[0], cmap='Blues')
    axes[0].set_title("Logistic Regression", fontweight='bold')
    
    cm_xgb = confusion_matrix(y_test, pred_xgb)
    ConfusionMatrixDisplay(cm_xgb, display_labels=["Home Loss", "Home Win"]).plot(ax=axes[1], cmap='Greens')
    axes[1].set_title("XGBoost", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {REPORTS_PATH / 'confusion_matrices.png'}")
    plt.close()
    
    # ========== FEATURE IMPORTANCE (XGB) ==========
    print("🌳 Generating feature importance plot...")
    importance = xgb.feature_importances_
    indices = np.argsort(importance)[-10:]  # Top 10
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(indices)), importance[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([FEATURES[i] for i in indices])
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title("XGBoost Feature Importance (Top 10)", fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "feature_importance.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {REPORTS_PATH / 'feature_importance.png'}")
    plt.close()
    
    # ========== METRICS COMPARISON TABLE ==========
    print("📋 Generating metrics comparison table...")
    metrics = {
        "Model": ["Logistic Regression", "XGBoost"],
        "Log Loss": [
            log_loss(y_test, p_logreg),
            log_loss(y_test, p_xgb)
        ],
        "Brier Score": [
            brier_score_loss(y_test, p_logreg),
            brier_score_loss(y_test, p_xgb)
        ],
        "AUC": [
            roc_auc_score(y_test, p_logreg),
            roc_auc_score(y_test, p_xgb)
        ],
        "Accuracy (@ 0.5)": [
            (pred_logreg == y_test).mean(),
            (pred_xgb == y_test).mean()
        ]
    }
    
    metrics_df = pd.DataFrame(metrics)
    print("\n" + "="*70)
    print("MODEL EVALUATION METRICS (Test Set)")
    print("="*70)
    print(metrics_df.to_string(index=False))
    print("="*70 + "\n")
    
    # Save metrics table as CSV
    metrics_df.to_csv(REPORTS_PATH / "metrics_comparison.csv", index=False)
    print(f"  ✅ Saved: {REPORTS_PATH / 'metrics_comparison.csv'}")
    
    print(f"\n✅ All evaluations complete. Reports saved to {REPORTS_PATH}")


if __name__ == "__main__":
    main()
