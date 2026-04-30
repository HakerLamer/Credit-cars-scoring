"""Train credit card default prediction models (v1 and v2)."""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import joblib
import json

BASE = os.path.dirname(os.path.abspath(__file__))

# Load data
df = pd.read_csv(os.path.join(BASE, '../data/UCI_Credit_Card.csv'))
df = df.drop(columns=['ID'])
df.columns = [c.replace('.', '_') for c in df.columns]

X = df.drop(columns=['default_payment_next_month'])
Y = df['default_payment_next_month']
FEATURES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# --- Model v1: Logistic Regression ---
pipeline_v1 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])
pipeline_v1.fit(X_train, y_train)
y_pred_v1 = pipeline_v1.predict(X_test)

print("=== Model v1: Logistic Regression ===")
print(classification_report(y_test, y_pred_v1))
metrics_v1 = {
    "f1": round(f1_score(y_test, y_pred_v1), 4),
    "precision": round(precision_score(y_test, y_pred_v1), 4),
    "recall": round(recall_score(y_test, y_pred_v1), 4)
}

# --- Model v2: Gradient Boosting ---
pipeline_v2 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4))
])
pipeline_v2.fit(X_train, y_train)
y_pred_v2 = pipeline_v2.predict(X_test)

print("\n=== Model v2: Gradient Boosting ===")
print(classification_report(y_test, y_pred_v2))
metrics_v2 = {
    "f1": round(f1_score(y_test, y_pred_v2), 4),
    "precision": round(precision_score(y_test, y_pred_v2), 4),
    "recall": round(recall_score(y_test, y_pred_v2), 4)
}

# Save models and metadata
joblib.dump(pipeline_v1, os.path.join(BASE, 'model_v1.pkl'))
joblib.dump(pipeline_v2, os.path.join(BASE, 'model_v2.pkl'))

meta = {
    "features": FEATURES,
    "v1_metrics": metrics_v1,
    "v2_metrics": metrics_v2
}
with open(os.path.join(BASE, 'model_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print("\nModels saved: model_v1.pkl, model_v2.pkl")
print(f"v1 metrics: {metrics_v1}")
print(f"v2 metrics: {metrics_v2}")
