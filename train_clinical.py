"""
train_clinical.py — Logistic Regression Only
=============================================
Trains on lung_cancer.csv, saves:
  models/clinical_model.pkl
  models/scaler.pkl
  models/label_encoder.pkl
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
data = pd.read_csv("dataset/lung_cancer.csv")
print(f"Dataset shape : {data.shape}")
print(f"Columns       : {list(data.columns)}\n")

X = data.iloc[:, :-1].copy()
y = data.iloc[:, -1].copy()

# ── ENCODE STRING COLUMNS ─────────────────────────────────────────────────────
for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y.astype(str))
print(f"Target classes      : {list(le_target.classes_)}")
print(f"Class distribution  : {np.bincount(y_encoded)}")
pickle.dump(le_target, open("models/label_encoder.pkl", "wb"))

# ── FEATURE ENGINEERING (fixed constants, no leakage) ─────────────────────────
X_eng = X.copy()

age_col     = next((c for c in X_eng.columns if c.strip().lower() == "age"), None)
smoking_col = next((c for c in X_eng.columns if "smok" in c.strip().lower()), None)
alcohol_col = next((c for c in X_eng.columns if "alcohol" in c.strip().lower()
                    or "drink" in c.strip().lower()), None)
genetic_col = next((c for c in X_eng.columns if "genetic" in c.strip().lower()
                    or "risk" in c.strip().lower()), None)

S_MIN, S_MAX = 1, 10

if age_col:
    X_eng["age_group"] = pd.cut(
        X_eng[age_col], bins=[0, 35, 55, 120], labels=[0, 1, 2]
    ).astype(float).fillna(1)

if smoking_col:
    X_eng["smoking_risk"] = (
        (X_eng[smoking_col].clip(S_MIN, S_MAX) - S_MIN) / (S_MAX - S_MIN)
    ) ** 2

if age_col and smoking_col:
    X_eng["age_x_smoking"] = X_eng[age_col] * X_eng[smoking_col]

if smoking_col and alcohol_col:
    X_eng["smoking_x_alcohol"] = X_eng[smoking_col] * X_eng[alcohol_col]

risk_cols = [c for c in [smoking_col, alcohol_col, genetic_col] if c]
if risk_cols:
    X_eng["total_risk"] = X_eng[risk_cols].sum(axis=1)

print(f"\nFeatures: {X.shape[1]} -> {X_eng.shape[1]} (after engineering)")

X_final = X_eng.values

# ── SPLIT first, then SCALE ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)

# ── TRAIN LOGISTIC REGRESSION ─────────────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING: Logistic Regression")
print("="*50)

model = LogisticRegression(
    C            = 1.0,
    max_iter     = 2000,
    solver       = "lbfgs",
    class_weight = "balanced",
    random_state = 42,
    multi_class  = "auto"
)
model.fit(X_train_scaled, y_train)

# ── CROSS-VALIDATION ──────────────────────────────────────────────────────────
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train,
                            cv=cv, scoring="f1_weighted")
print(f"CV F1 (weighted): {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

# ── EVALUATE ──────────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print(f"F1-Score      : {f1*100:.2f}%")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
pickle.dump(model,  open("models/clinical_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl",         "wb"))

print("\nclinical_model.pkl  saved")
print("scaler.pkl          saved")
print("label_encoder.pkl   saved")
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print(f"CV F1-Score   : {cv_scores.mean()*100:.2f}%")