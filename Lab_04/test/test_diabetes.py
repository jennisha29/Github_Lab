import sys
import subprocess
from pathlib import Path
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "diabetes_model.pkl"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from main import app 
client = TestClient(app)
SAMPLE = {
    "age": 0.038,
    "sex": 0.051,
    "bmi": 0.062,
    "bp": 0.022,
    "s1": -0.044,
    "s2": -0.034,
    "s3": -0.043,
    "s4": -0.002,
    "s5": 0.019,
    "s6": -0.018,
}

@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    if not MODEL_PATH.exists():
        subprocess.run(
            [sys.executable, "train.py"],
            cwd=PROJECT_ROOT / "src",
            check=True,
        )

def test_load_data_shape():
    from data import load_data

    X, y = load_data()
    assert X.shape == (442, 10)
    assert y.shape[0] == 442

def test_load_data_no_nulls():
    from data import load_data

    X, y = load_data()
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()

def test_load_data_target_range():
    from data import load_data

    _, y = load_data()
    assert y.min() >= 25
    assert y.max() <= 346

def test_load_data_normalised():
    from data import load_data

    X, _ = load_data()
    assert abs(X.mean()) < 0.01

def test_split_80_20_ratio():
    from data import load_data, split_data

    X, y = load_data()
    _, X_test, _, _ = split_data(X, y)
    assert abs(len(X_test) / len(X) - 0.2) < 0.02

def test_split_same_result_each_time():
    from data import load_data, split_data

    X, y = load_data()
    assert np.array_equal(split_data(X, y)[0], split_data(X, y)[0])

def test_feature_names_correct():
    from data import get_feature_names

    assert set(get_feature_names()) == {
        "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"
    }

def test_descriptions_cover_all_features():
    from data import get_feature_names, get_feature_descriptions

    descs = get_feature_descriptions()
    for name in get_feature_names():
        assert name in descs

def test_descriptions_not_empty():
    from data import get_feature_descriptions

    for val in get_feature_descriptions().values():
        assert isinstance(val, str)
        assert len(val) > 0

def test_fit_model_creates_pkl(tmp_path, monkeypatch):
    from data import load_data, split_data
    import train as t

    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)

    out = tmp_path / "m.pkl"
    monkeypatch.setattr(t, "MODEL_PATH", out)
    t.fit_model(X_train, y_train)

    assert out.exists()

def test_fit_model_pkl_loadable(tmp_path, monkeypatch):
    from data import load_data, split_data
    import train as t

    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)

    out = tmp_path / "m.pkl"
    monkeypatch.setattr(t, "MODEL_PATH", out)
    t.fit_model(X_train, y_train)

    assert hasattr(joblib.load(out), "predict")

def test_fit_model_good_accuracy(tmp_path, monkeypatch):
    from data import load_data, split_data
    import train as t

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    out = tmp_path / "m.pkl"
    monkeypatch.setattr(t, "MODEL_PATH", out)
    t.fit_model(X_train, y_train)

    preds = joblib.load(out).predict(X_test)
    assert r2_score(y_test, preds) > 0.35
    assert np.sqrt(mean_squared_error(y_test, preds)) < 65

def test_predict_single_row():
    from predict import predict_data

    row = [[0.038, 0.051, 0.062, 0.022, -0.044, -0.034, -0.043, -0.002, 0.019, -0.018]]
    assert len(predict_data(row)) == 1

def test_predict_batch_size():
    from predict import predict_data
    from data import load_data, split_data

    X, y = load_data()
    _, X_test, _, _ = split_data(X, y)
    assert len(predict_data(X_test)) == len(X_test)

def test_predict_finite_scores():
    from predict import predict_data
    from data import load_data, split_data

    X, y = load_data()
    _, X_test, _, _ = split_data(X, y)
    assert np.isfinite(predict_data(X_test)).all()

def test_predict_missing_model_raises(tmp_path, monkeypatch):
    import predict as p

    monkeypatch.setattr(p, "MODEL_PATH", tmp_path / "none.pkl")
    with pytest.raises(FileNotFoundError):
        p.predict_data([[0] * 10])

def test_model_info_correct_values():
    from predict import get_model_info

    info = get_model_info()
    assert info["model_type"] == "Ridge"
    assert info["alpha"] == 1.0
    assert info["n_features"] == 10
    assert np.isfinite(info["intercept"])


@pytest.mark.parametrize(
    "score,expected",
    [
        (25.0, "Low"),
        (131.9, "Low"),
        (132.0, "Medium"),
        (212.9, "Medium"),
        (213.0, "High"),
        (346.0, "High"),
    ],
)
def test_score_to_risk_bands(score, expected):
    from main import score_to_risk

    assert score_to_risk(score) == expected

def test_to_features_correct_shape():
    from main import to_features, DiabetesData

    result = to_features(DiabetesData(**SAMPLE))
    assert isinstance(result, list)
    assert len(result[0]) == 10

def test_to_features_correct_values():
    from main import to_features, DiabetesData

    row = to_features(DiabetesData(**SAMPLE))[0]
    assert row[0] == pytest.approx(SAMPLE["age"])
    assert row[2] == pytest.approx(SAMPLE["bmi"])
    assert row[-1] == pytest.approx(SAMPLE["s6"])

def test_health_check_ok():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}

def test_predict_valid_input():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_score" in data
    assert data["predicted_score"] > 0
    assert data["risk_level"] in {"Low", "Medium", "High"}

def test_predict_missing_fields():
    r = client.post("/predict", json={"age": 0.038, "sex": 0.051})
    assert r.status_code == 422

def test_predict_wrong_type():
    r = client.post("/predict", json={**SAMPLE, "age": "bad"})
    assert r.status_code == 422

def test_predict_same_result():
    r1 = client.post("/predict", json=SAMPLE).json()["predicted_score"]
    r2 = client.post("/predict", json=SAMPLE).json()["predicted_score"]
    assert r1 == r2

def test_features_returns_10_with_names():
    r = client.get("/features")
    assert r.status_code == 200
    features = r.json()["features"]
    assert len(features) == 10
    assert all("name" in f and "description" in f for f in features)

def test_risk_bands_no_gaps():
    r = client.get("/risk-bands")
    assert r.status_code == 200
    bands = r.json()["risk_bands"]
    assert set(bands.keys()) == {"Low", "Medium", "High"}
    assert bands["Low"]["max"] + 1 == bands["Medium"]["min"]
    assert bands["Medium"]["max"] + 1 == bands["High"]["min"]
    assert bands["Low"]["min"] == 25
    assert bands["High"]["max"] == 346

def test_model_info_endpoint():
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_type"] == "Ridge"
    assert data["alpha"] == 1.0
    assert data["n_features"] == 10