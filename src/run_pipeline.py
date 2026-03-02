from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

STEPS = [
    "fetch_nhl_games_api.py",
    "phase2_load_and_clean.py",
    "phase3_make_features.py",
    "phase4_train_model.py",
    "phase5_build_team_state.py",
]


def run_step(script_name: str) -> None:
    script_path = SRC / script_name
    print(f"\n▶ Running {script_name}")
    subprocess.run([sys.executable, str(script_path)], cwd=ROOT, check=True)


def main() -> None:
    for step in STEPS:
        run_step(step)
    print("\n✅ Pipeline complete.")
    print("Artifacts:")
    print(f"- {ROOT / 'data/processed/games_clean.csv'}")
    print(f"- {ROOT / 'data/processed/games_with_features.csv'}")
    print(f"- {ROOT / 'models/logreg_moneyline.joblib'}")
    print(f"- {ROOT / 'data/processed/team_state_latest.csv'}")


if __name__ == "__main__":
    main()
