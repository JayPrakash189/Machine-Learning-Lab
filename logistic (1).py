# LOGISTIC REGRESSION MODEL

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Load dataset
iris_data = load_iris()
features = iris_data.data
targets = iris_data.target

# Convert to binary classification (class 0 vs others)
targets = (targets == 0).astype(int)

# Split dataset
train_features, test_features, train_targets, test_targets = train_test_split(
    features, targets, test_size=0.2, random_state=42
)

# Create model
log_reg_model = LogisticRegression()

# Train model
log_reg_model.fit(train_features, train_targets)

# Predict on test data
predicted_targets = log_reg_model.predict(test_features)

# Evaluation
print("Accuracy:", accuracy_score(test_targets, predicted_targets))
print("\nClassification Report:\n", classification_report(test_targets, predicted_targets))
