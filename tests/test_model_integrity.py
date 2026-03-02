"""
Test model integrity: loading, predictions, feature count.
"""
import joblib
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def model_path():
    """Path to trained model."""
    return Path(__file__).parent.parent / "models" / "logreg_moneyline.joblib"


@pytest.fixture
def model_dict(model_path):
    """Load trained model dictionary."""
    assert model_path.exists(), f"Model file not found: {model_path}"
    return joblib.load(model_path)


@pytest.fixture
def model(model_dict):
    """Extract logreg model from dictionary."""
    assert isinstance(model_dict, dict), "Model should be a dictionary"
    assert "logreg" in model_dict, "Dictionary should contain 'logreg' model"
    return model_dict["logreg"]


def test_model_loads_successfully(model_dict):
    """Verify model artifact loads without errors."""
    assert model_dict is not None
    assert isinstance(model_dict, dict)
    assert "logreg" in model_dict or "xgb" in model_dict


def test_model_has_required_attributes(model):
    """Verify model has expected structure."""
    # Check for pipeline structure (should have predict_proba)
    assert hasattr(model, "predict_proba"), "Model missing predict_proba method"
    assert hasattr(model, "predict"), "Model missing predict method"


def test_model_features_defined(model_dict):
    """Verify feature list is defined in model dict."""
    assert "features" in model_dict, "Model dict should contain 'features' list"
    features = model_dict["features"]
    
    # Should have multiple features
    assert len(features) > 5, f"Expected 5+ features, got {len(features)}"


def test_model_prediction_format(model, model_dict):
    """Verify model predictions are valid probabilities."""
    # Create synthetic feature vector using actual model features
    num_features = len(model_dict["features"])
    X = np.random.randn(1, num_features)
    
    # Model should return probabilities in [0, 1]
    preds = model.predict_proba(X)
    
    assert preds.shape[0] == 1, "Wrong number of predictions"
    assert len(preds[0]) == 2, "Should output 2 class probabilities"
    
    # Both probabilities should sum to 1
    assert np.isclose(preds[0].sum(), 1.0), "Probabilities don't sum to 1"
    
    # Each probability should be in [0, 1]
    assert (preds[0] >= 0).all() and (preds[0] <= 1).all(), "Probability outside [0, 1]"


def test_model_expected_feature_count(model, model_dict):
    """Verify model expects correct number of features."""
    expected_features = len(model_dict["features"])
    
    # Get feature count from model pipeline
    if hasattr(model, "named_steps"):
        # It's a pipeline - check the final estimator
        final_step = model.named_steps.get("clf", model)
        if hasattr(final_step, "n_features_in_"):
            assert final_step.n_features_in_ == expected_features, \
                f"Expected {expected_features} features, model has {final_step.n_features_in_}"


def test_model_reproducible_predictions(model, model_dict):
    """Verify model produces consistent predictions."""
    np.random.seed(42)
    X = np.random.randn(3, len(model_dict["features"]))
    
    pred1 = model.predict_proba(X)
    pred2 = model.predict_proba(X)
    
    assert np.allclose(pred1, pred2), "Model predictions not reproducible"


def test_xgb_model_present(model_dict):
    """Verify XGBoost model is also available."""
    assert "xgb" in model_dict, "Dictionary should contain 'xgb' model"
    xgb_model = model_dict["xgb"]
    assert hasattr(xgb_model, "predict_proba"), "XGBoost model missing predict_proba"

