# Lab_05 — Experiment Tracking with Weights & Biases

---

## Overview

In this lab, I used **Weights & Biases (W&B)** to track my machine learning experiments.
Instead of just printing results in the notebook, I logged everything — metrics, charts, models —
to a W&B dashboard where I can visualize training, compare runs, and version my models.

- **Lab1** — a traditional ML model (XGBoost) on tabular data
- **Lab2** — a deep learning model (CNN with Keras) on image data

Both notebooks log hyperparameters, training progress, evaluation metrics, and visualizations
to the same W&B project so I can compare them side by side.

---

## Lab 1 — XGBoost on Letter Recognition

### What I did

I downloaded the UCI Letter Recognition dataset (20,000 samples, 16 features)
and trained an XGBoost classifier to recognize uppercase letters A–Z (26 classes).
Everything is logged to W&B.

### Key features

- Stratified train / val / test split (70/15/15)
- XGBoost with `multi:softprob` objective and L1/L2 regularization
- Early stopping on validation loss (patience 20, up to 300 rounds)
- I log accuracy, precision, recall, F1, error rate to W&B
- I log a confusion matrix and feature importance bar chart
- I print a per-class classification report
- I save the trained model as a W&B artifact

### Results

| Metric         | Value     |
| -------------- | --------- |
| Accuracy       | 96.1%     |
| Precision      | 96.2%     |
| Recall         | 96.1%     |
| F1             | 96.1%     |
| Error Rate     | 3.9%      |
| Best Iteration | 299 / 300 |

### Screenshots

#### Run Summary

This shows the final metrics and training logs — dataset loaded, split sizes, loss decreasing over 300 rounds, and final accuracy/precision/recall/F1 on the held-out test set.

![Run summary](screenshots/lab1_run_summary.png)

#### Loss Curves

This shows how both training loss and validation loss decrease over time. The curves stay close together, meaning the model is learning well without overfitting.

![Loss curves](screenshots/lab1_loss_curves.png)

#### Feature Importance

This shows which of the 16 features matter most for predicting letters. Features like `x-ege`, `xy2br`, and `y2bar` contribute the most.

![Feature importance](screenshots/lab1_feature_importance.png)

#### Confusion Matrix

A 26×26 grid showing how well the model predicts each letter. Dark blue along the diagonal means correct predictions. Very few off-diagonal errors at 96% accuracy.

![Confusion matrix](screenshots/lab1_confusion_matrix.png)

---

## Lab 2 — CNN on CIFAR-100

### What I did

I loaded the CIFAR-100 dataset with coarse labels (32×32 RGB images, 20 superclasses like
"flowers", "fish", "vehicles", "insects"), trained a deep CNN, and logged everything to W&B.

### Key features

- CIFAR-100 with 20 superclasses (coarse labels)
- 3 convolutional blocks with BatchNormalization and Dropout
- GlobalAveragePooling instead of Flatten
- Adam optimizer with ReduceLROnPlateau
- Data augmentation (rotation, shifts, horizontal flip)
- Early stopping (patience 5, restores best weights)
- Stratified train / val / test split — val for callbacks, test only at the end
- I log loss curves, sample predictions with images, confusion matrix, per-class accuracy
- I print a full classification report
- I save the trained model as a W&B artifact

### Results

| Metric     | Value   |
| ---------- | ------- |
| Accuracy   | 37.3%   |
| Precision  | 41.5%   |
| Recall     | 37.3%   |
| F1         | 36.5%   |
| Best Epoch | 13 / 15 |

37% is expected for a baseline CNN on CIFAR-100 with only 15k training samples.
Random guessing on 20 classes would give 5%, so my model performs 7× better than chance.
Training is stable with no overfitting — the focus of this lab is experiment tracking, not achieving state-of-the-art accuracy.

**Best classes:** flowers (0.58 F1), fruit_vegetables (0.50 F1), large_outdoor_natural (0.56 F1)

**Hardest classes:** non-insect_invertebrates (0.07 F1), food_containers (0.19 F1)

### Screenshots

#### Training Output

This shows the W&B run syncing, dataset split sizes, epoch-by-epoch training progress with accuracy and loss, and the ReduceLROnPlateau kicking in at epoch 5.

![Training output](screenshots/lab2_training_output.png)

#### Run Summary

This shows the W&B run summary with final batch and epoch metrics — accuracy, loss, learning rate, and validation scores logged automatically.

![Run summary](screenshots/lab2_run_summary.png)

#### Classification Report

This shows final test metrics (accuracy, precision, recall, F1) and a per-class breakdown showing which superclasses the model handles well and which it struggles with.

![Classification report](screenshots/lab2_classification_report.png)

---

## Project Structure

```
Lab_05/
├── Lab1.ipynb                  # xgboost on letter recognition
├── Lab2.ipynb                  # cnn on cifar-100
├── README.md
├── SETUP.md                    # how to set up and run
├── screenshots/
│   ├── lab1_run_summary.png
│   ├── lab1_loss_curves.png
│   ├── lab1_feature_importance.png
│   ├── lab1_confusion_matrix.png
│   ├── lab2_training_output.png
│   ├── lab2_run_summary.png
│   └── lab2_classification_report.png
├── artifacts/                  # created at runtime
└── wandb/                      # created at runtime
```

---

## W&B Project Links

- **Project:** [Lab5](https://wandb.ai/mlops-team-northeastern-university/Lab5)
- **Lab1 Run:** [xgboost_letter_recognition](https://wandb.ai/mlops-team-northeastern-university/Lab5/runs/medn5pdy)
- **Lab2 Run:** [cifar100_deep_cnn](https://wandb.ai/mlops-team-northeastern-university/Lab5/runs/6y8qr4ru)
