# Lab_04 — FastAPI + Docker + Frontend

A machine learning API that trains a **Ridge Regression** model on the sklearn
Diabetes dataset to predict disease progression scores, through **FastAPI**
and with a custom **DIABETIX** frontend dashboard.

---

## Overview

Loading the sklearn Diabetes dataset (442 patients, 10 normalised features),
training a regression model, and serving it as a live REST API with a
browser based dashboard:

- `diabetes_model.pkl` — the trained Ridge Regression model
- `POST /predict` — returns a progression score (25–346) and risk level
- `frontend/index.html` — DIABETIX dashboard UI

---

## Project Structure

```
Lab_04/
├── src/
│   ├── data.py               # loads and splits the Diabetes dataset
│   ├── train.py              # trains Ridge Regression, saves .pkl
│   ├── predict.py            # loads model, runs inference
│   └── main.py               # FastAPI app — 5 endpoints + CORS
├── test/
│   └── test_diabetes.py      # 33 unit tests
├── frontend/
│   ├── index.html            # DIABETIX dashboard
│   └── nginx.conf            # serves UI
├── model/
│   └── diabetes_model.pkl
├── assets/
│   ├── docs.png
│   ├── api_response.png
│   └── interface.png
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── README.md
└── SETUP_README.md           # step-by-step run instructions
```

---

## Results

| Metric  | Value                                           |
| ------- | ----------------------------------------------- |
| R²      | 0.4192 — model explains ~42% of score variation |
| RMSE    | 55.47                                           |
| Tests   | 33 / 33 passed                                  |
| Model   | Ridge Regression (alpha = 1.0)                  |
| Dataset | sklearn Diabetes (442 samples, 10 features)     |

---

## API Endpoints

| Method | Endpoint      | Description                          |
| ------ | ------------- | ------------------------------------ |
| GET    | `/`           | Health check                         |
| POST   | `/predict`    | Returns predicted score + risk level |
| GET    | `/features`   | Lists all 10 input features          |
| GET    | `/risk-bands` | Shows Low / Medium / High thresholds |
| GET    | `/model-info` | Model type, alpha, intercept         |

---

## Frontend

After running the containers, open the dashboard:

```bash
docker compose up --build
```

Then open: `http://localhost`
