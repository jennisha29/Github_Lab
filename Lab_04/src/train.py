from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from data import load_data, split_data
import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "diabetes_model.pkl"

def fit_model(X_train, y_train):
    """
    Train a Ridge Regression model and save it to disk.

    Ridge regression applies L2 regularisation to reduce overfitting,
    making it well-suited for this small (442-sample) dataset.

    Args:
        X_train (numpy.ndarray): Training features — 10 normalised measurements.
        y_train (numpy.ndarray): Continuous disease progression scores.
    """
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    joblib.dump(ridge, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    print(f"Test RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"Test R²   : {r2_score(y_test, y_pred):.4f}")