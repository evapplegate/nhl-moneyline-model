"""
Test data pipeline outputs for consistency and correctness.
"""
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def data_dir():
    """Path to processed data directory."""
    return Path(__file__).parent.parent / "data" / "processed"


def test_games_clean_schema(data_dir):
    """Verify games_clean.csv has correct schema."""
    df = pd.read_csv(data_dir / "games_clean.csv")
    
    expected_cols = {"date", "home_team", "away_team", "home_goals", "away_goals", "home_win"}
    assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"
    
    # Check data types
    assert pd.api.types.is_integer_dtype(df["home_goals"])
    assert pd.api.types.is_integer_dtype(df["away_goals"])
    assert pd.api.types.is_integer_dtype(df["home_win"])


def test_games_clean_no_nulls(data_dir):
    """Verify no null values in critical columns."""
    df = pd.read_csv(data_dir / "games_clean.csv")
    
    critical_cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "home_win"]
    for col in critical_cols:
        assert df[col].isnull().sum() == 0, f"Column '{col}' has null values"


def test_games_clean_size(data_dir):
    """Verify games_clean.csv has expected number of rows."""
    df = pd.read_csv(data_dir / "games_clean.csv")
    
    # Should have ~6500+ games (Oct 2021 - Mar 2026)
    assert len(df) >= 6500, f"Expected 6500+ games, got {len(df)}"


def test_home_win_baseline(data_dir):
    """Verify home team win rate is ~53% (known baseline)."""
    df = pd.read_csv(data_dir / "games_clean.csv")
    
    home_win_rate = df["home_win"].mean()
    
    # Home advantage typically ~53% in NHL
    assert 0.50 < home_win_rate < 0.56, f"Home win rate {home_win_rate:.2%} outside expected range"


def test_games_with_features_schema(data_dir):
    """Verify games_with_features.csv has all required feature columns."""
    df = pd.read_csv(data_dir / "games_with_features.csv")
    
    required_features = {
        "elo_diff",
        "home_rolling_win_pct",
        "away_rolling_win_pct",
        "home_rolling_goal_diff",
        "away_rolling_goal_diff",
        "rest_diff",
    }
    
    assert required_features.issubset(df.columns), f"Missing features: {required_features - set(df.columns)}"


def test_games_with_features_no_nan_in_features(data_dir):
    """Verify minimal NaN values in core ML features."""
    df = pd.read_csv(data_dir / "games_with_features.csv")
    
    feature_cols = [
        "elo_diff",
        "home_rolling_win_pct",
        "away_rolling_win_pct",
        "home_rolling_goal_diff",
        "away_rolling_goal_diff",
        "rest_diff",
    ]
    
    nan_counts = df[feature_cols].isnull().sum()
    
    # Allow some NaNs from initial games (typically 1-5)
    # but they should be minimal
    total_nans = nan_counts.sum()
    max_acceptable = 5
    
    assert total_nans <= max_acceptable, \
        f"Too many NaNs in features ({total_nans}): {nan_counts[nan_counts > 0].to_dict()}"



def test_team_state_latest_schema(data_dir):
    """Verify team_state_latest.csv has required team columns."""
    df = pd.read_csv(data_dir / "team_state_latest.csv")
    
    # Check for team column (using 'team' not 'team_code')
    assert "team" in df.columns
    
    # Should have ELO rating
    assert any(col in df.columns for col in ["current_elo", "elo"]), "Missing ELO column"
    
    # Should have rolling stats
    assert any(col in df.columns for col in ["home_rolling_win_pct", "away_rolling_win_pct"]), \
        "Missing rolling win pct columns"
    
    # Should have ~37 teams (including 2025-26 expansion)
    assert len(df) >= 30, f"Expected 30+ teams, got {len(df)}"



def test_team_state_latest_no_nulls(data_dir):
    """Verify no null values in team state data."""
    df = pd.read_csv(data_dir / "team_state_latest.csv")
    
    assert df.isnull().sum().sum() == 0, "Found null values in team state"


def test_upcoming_predictions_exists(data_dir):
    """Verify upcoming predictions CSV exists (if generated)."""
    pred_file = data_dir / "upcoming_predictions.csv"
    
    if pred_file.exists():
        df = pd.read_csv(pred_file)
        
        # Should have prediction columns
        assert "p_home_win" in df.columns
        assert len(df) > 0
        
        # Probabilities should be in [0, 1]
        assert (df["p_home_win"] >= 0).all() and (df["p_home_win"] <= 1).all()

