import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Diabetes dataset and return features and target values.

    The dataset contains 10 baseline variables (age, sex, bmi, blood pressure,
    and 6 serum measurements) for 442 diabetes patients. The target is a
    quantitative measure of disease progression one year after baseline.

    Returns:
        X (numpy.ndarray): Feature matrix of shape (442, 10).
        y (numpy.ndarray): Continuous target values ranging from 25 to 346.
    """
    diabetes = load_diabetes()
    return diabetes.data, diabetes.target

def split_data(X, y):
    """
    Split data into training (80%) and testing (20%) sets.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target values.

    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_feature_names():
    """Return the list of feature names for the Diabetes dataset."""
    return list(load_diabetes().feature_names)

def get_feature_descriptions():
    """Return human-readable descriptions for each feature."""
    return {
        "age": "Age (normalised)",
        "sex": "Sex (normalised)",
        "bmi": "Body Mass Index (normalised)",
        "bp":  "Average Blood Pressure (normalised)",
        "s1":  "Total serum cholesterol (normalised)",
        "s2":  "LDL cholesterol (normalised)",
        "s3":  "HDL cholesterol (normalised)",
        "s4":  "Total cholesterol / HDL ratio (normalised)",
        "s5":  "Log of serum triglycerides (normalised)",
        "s6":  "Blood sugar level (normalised)",
    }