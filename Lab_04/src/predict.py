import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "diabetes_model.pkl"

def _load_model():
    """Load and return the trained Ridge Regression model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)

def predict_data(X):
    """
    Predict disease progression scores for the given input features.

    Args:
        X (array-like): 2D array of shape (n_samples, 10).

    Returns:
        y_pred (numpy.ndarray): Predicted scores roughly in range 25 - 346.
    """
    return _load_model().predict(X)

def get_model_info():
    """
    Return metadata about the loaded Ridge Regression model.

    Returns:
        dict: model_type, alpha, n_features, intercept.
    """
    model = _load_model()
    return {
        "model_type": type(model).__name__,
        "alpha":      model.alpha,
        "n_features": model.n_features_in_,
        "intercept":  round(float(model.intercept_), 4),
    }