"""
Test FastAPI endpoints: health, teams, predictions.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add app directory to path
app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_path))

from main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_teams_endpoint(client):
    """Verify /teams endpoint returns available teams."""
    response = client.get("/teams")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should return a list of team codes
    assert "teams" in data or isinstance(data, list)
    
    teams = data.get("teams", data) if isinstance(data, dict) else data
    
    # Should have teams
    assert len(teams) >= 30, f"Expected 30+ teams, got {len(teams)}"
    
    # All teams should be 3-letter codes
    assert all(isinstance(t, str) and len(t) == 3 for t in teams)


def test_predict_endpoint_exists(client):
    """Verify /predict endpoint is accessible."""
    # Test that the endpoint exists by checking for 422 (validation error)
    # when sending empty payload (not 404 for missing endpoint)
    
    response = client.post("/predict", json={})
    
    # Should get 422 for validation error, not 404 for missing endpoint
    assert response.status_code in [422, 400], \
        f"Expected 422 (validation), got {response.status_code}"


def test_predict_invalid_team(client):
    """Verify /predict endpoint handles invalid teams."""
    payload = {
        "game_date": "2026-03-10",
        "home_team": "INVALID",
        "away_team": "NYR",
        "model": "logreg",
    }
    
    response = client.post("/predict", json=payload)
    
    # Should return 404 or 400 for invalid team
    assert response.status_code in [400, 404, 422]


def test_predict_missing_required_field(client):
    """Verify /predict endpoint requires all fields."""
    # Missing home_team
    payload = {
        "game_date": "2026-03-10",
        "away_team": "NYR",
    }
    
    response = client.post("/predict", json=payload)
    
    # Should return 422 (validation error)
    assert response.status_code == 422


def test_api_documentation_availability(client):
    """Verify OpenAPI documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/openapi.json")
    assert response.status_code == 200

