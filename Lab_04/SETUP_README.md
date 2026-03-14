# Lab_04

---

## 1. Prerequisites

Docker Desktop must be installed and running.
Download from: `https://www.docker.com/get-started`

Check it is working:

```bash
docker --version
```

---

## 2. Navigate to the right directory

```bash
cd Lab_04
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the Model

```bash
cd src
python train.py
cd ..
```

Expected:

```
Model saved to .../Lab_04/model/diabetes_model.pkl
Test RMSE : 55.4745
Test R²   : 0.4192
```

---

## 5. Run the Tests

```bash
pytest test/test_diabetes.py -v
```

Expected:

```
33 passed in ~Xs
```

---

## 6. Build and Start Docker

```bash
docker compose up --build
```

Starts two containers:

| Container           | Role                     | Port            |
| ------------------- | ------------------------ | --------------- |
| `diabetes-api`      | FastAPI + Ridge model    | 8000 (internal) |
| `diabetes-frontend` | nginx serving UI + proxy | 80              |

---

## 7. Open in Browser

| URL                | What you see      |
| ------------------ | ----------------- |
| `http://localhost` | DIABETIX frontend |

---

## 8. Test the API

```bash
curl http://localhost/api/
```

Expected:

```json
{ "status": "healthy" }
```

Run a prediction:

```bash
curl -X POST http://localhost/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.038, "sex": 0.051, "bmi": 0.062, "bp": 0.022,
    "s1": -0.044, "s2": -0.034, "s3": -0.043,
    "s4": -0.002, "s5": 0.019, "s6": -0.018
  }'
```

Expected:

```json
{
  "predicted_score": 181.82,
  "risk_level": "Medium"
}
```

---

## 9. Stop the Containers

```bash
docker compose down
```

---

## Common Commands

```bash
cd Lab_04

# train model
cd src && python train.py && cd ..

# run tests
pytest test/test_diabetes.py -v

# start full stack
docker compose up --build

# stop
docker compose down

# health check
curl http://localhost/api/

# predict
curl -X POST http://localhost/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.038,"sex":0.051,"bmi":0.062,"bp":0.022,"s1":-0.044,"s2":-0.034,"s3":-0.043,"s4":-0.002,"s5":0.019,"s6":-0.018}'
```

---

## Sample Input Values

| Feature | Value  | Description                  |
| ------- | ------ | ---------------------------- |
| age     | 0.038  | Age (normalised)             |
| sex     | 0.051  | Sex (normalised)             |
| bmi     | 0.062  | Body Mass Index (normalised) |
| bp      | 0.022  | Avg Blood Pressure           |
| s1      | -0.044 | Total serum cholesterol      |
| s2      | -0.034 | LDL cholesterol              |
| s3      | -0.043 | HDL cholesterol              |
| s4      | -0.002 | Chol / HDL ratio             |
| s5      | 0.019  | Log triglycerides            |
| s6      | -0.018 | Blood sugar level            |
