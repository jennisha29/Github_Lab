# Lab_05 — Setup

---

## 1. Prerequisites

- Python 3.10–3.12 (TensorFlow does not support 3.13+)
- [Weights & Biases](https://wandb.ai) account (free tier works)

Check Python version:

```bash
python3 --version
```

---

## 2. Navigate into Lab_05

```bash
cd Lab_05
```

---

## 3. Create and Activate Environment

**Mac / Linux:**

```bash
conda create -n mlops_lab5 python=3.12 -y
conda activate mlops_lab5
```

**Windows:**

```bash
conda create -n mlops_lab5 python=3.12 -y
conda activate mlops_lab5
```

Or use an existing environment:

```bash
conda activate mlops_project
```

---

## 4. Install Dependencies

**Windows:**

```bash
pip install wandb xgboost scikit-learn tensorflow numpy pandas pillow nbformat
```

**Mac (Apple Silicon):**

```bash
pip install wandb xgboost scikit-learn numpy pandas pillow nbformat
pip install tensorflow-macos
brew install libomp
```

---

## 5. Login to W&B

```bash
wandb login
```

Paste your API key from: `https://wandb.ai/authorize`

---

## 6. Run Lab1 (XGBoost)

### VS Code

1. Open `Lab1.ipynb`
2. Select kernel
3. If prompted to install `ipykernel`, click **Install**
4. Run all cells

### Jupyter Notebook

```bash
jupyter notebook
```

1. Open `Lab1.ipynb`
2. Kernel → Change Kernel
3. Run all cells

### Expected Output

```
Loaded 20000 samples, 16 features, 26 classes
Train: 14008 | Val: 2992 | Test: 3000

[0]     train-mlogloss:2.76541    val-mlogloss:2.76862
...
[299]   train-mlogloss:0.02916    val-mlogloss:0.13097

Best iteration: 299
Accuracy   = 0.9610
Precision  = 0.9615
Recall     = 0.9610
F1         = 0.9611
Error Rate = 0.0390
```

---

## 7. Run Lab2 (CNN)

Same steps as Lab1, but open `Lab2.ipynb` instead.

### Expected Output

```
Train: (12750, 32, 32, 3) | Val: (2250, 32, 32, 3) | Test: (5000, 32, 32, 3)
Epoch 1/15
...
Restoring model weights from the end of the best epoch: 13.

Test Loss      = 1.9883
Test Accuracy  = 0.3732
Test Precision = 0.4148
Test Recall    = 0.3732
Test F1        = 0.3654
```

---

## 8. Check W&B Dashboard

After each run finishes, visit:

```
https://wandb.ai/mlops-team-northeastern-university/Lab5
```

You should see loss curves, confusion matrix, feature importance, sample predictions, and model artifacts.

---

## 9. Verify Outputs

```bash
ls artifacts/
```

should display:

```
xgb_letter_model.json       ← trained xgboost model (Lab1)
cifar100_model.h5            ← trained cnn model (Lab2)
model_summary.txt            ← cnn architecture summary (Lab2)
```

## Common Commands

```bash
cd Lab_05
conda activate mlops_lab5
pip install wandb xgboost scikit-learn tensorflow numpy pandas pillow nbformat
wandb login
jupyter notebook
ls artifacts/
```
