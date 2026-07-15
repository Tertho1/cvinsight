"""
Week 5 — ML Text Classifier for CV Quality.

Pipeline:
  1. Load classifier_training_data.csv (raw_text + label from rubric)
  2. TF-IDF vectorize raw CV text
  3. Train Logistic Regression baseline + XGBoost
  4. Report accuracy, F1, confusion matrix
  5. Compare vs rubric (majority-class baseline + CV analysis)
  6. Save best model

Usage:
    python scripts/vectorize_cvs.py
"""

import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.getcwd())

DATA_DIR = "data/processed"
INPUT_CSV = os.path.join(DATA_DIR, "classifier_training_data.csv")
MODELS_DIR = "models"
RESULTS_FILE = os.path.join(DATA_DIR, "classifier_comparison.csv")


def train_models():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} CVs from {INPUT_CSV}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Score range: {df['total_score'].min():.0f} - {df['total_score'].max():.0f}")
    print(f"Score mean: {df['total_score'].mean():.1f}")

    texts = df["raw_text"].fillna("").tolist()
    labels_str = df["label"].tolist()

    # Encode labels for XGBoost
    le = LabelEncoder()
    labels_num = le.fit_transform(labels_str)
    print(f"Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels_num, test_size=0.2, random_state=42, stratify=labels_num
    )
    # Keep string versions for interpretability
    y_train_str = le.inverse_transform(y_train)
    y_test_str = le.inverse_transform(y_test)

    label_counts = Counter(le.inverse_transform(y_train))
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train label distribution: {dict(label_counts)}")

    # ==============================
    # Logistic Regression (TF-IDF)
    # ==============================
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION (TF-IDF)")
    print("=" * 60)

    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True,
            min_df=5, max_df=0.85
        )),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
            solver="lbfgs"
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr_num = lr_pipe.predict(X_test)
    y_pred_lr = le.inverse_transform(y_pred_lr_num)

    print("\nClassification Report:")
    print(classification_report(y_test_str, y_pred_lr, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_str, y_pred_lr))

    lr_f1 = f1_score(y_test_str, y_pred_lr, average="weighted")
    lr_acc = accuracy_score(y_test_str, y_pred_lr)

    # Save LR model
    os.makedirs(MODELS_DIR, exist_ok=True)
    import joblib
    lr_path = os.path.join(MODELS_DIR, "lr_baseline.pkl")
    joblib.dump(lr_pipe, lr_path)
    print(f"Saved: {lr_path}")

    # ==============================
    # XGBoost (TF-IDF)
    # ==============================
    print("\n" + "=" * 60)
    print("XGBOOST (TF-IDF)")
    print("=" * 60)

    from xgboost import XGBClassifier

    xgb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True,
            min_df=5, max_df=0.85
        )),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="multi:softmax", num_class=3,
            eval_metric="mlogloss", random_state=42,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    y_pred_xgb_num = xgb_pipe.predict(X_test)
    y_pred_xgb = le.inverse_transform(y_pred_xgb_num)

    print("\nClassification Report:")
    print(classification_report(y_test_str, y_pred_xgb, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_str, y_pred_xgb))

    xgb_f1 = f1_score(y_test_str, y_pred_xgb, average="weighted")
    xgb_acc = accuracy_score(y_test_str, y_pred_xgb)

    xgb_path = os.path.join(MODELS_DIR, "xgb_classifier.pkl")
    joblib.dump(xgb_pipe, xgb_path)
    print(f"Saved: {xgb_path}")

    # ==============================
    # Baseline: Majority Class
    # ==============================
    print("\n" + "=" * 60)
    print("BASELINE: MAJORITY CLASS")
    print("=" * 60)
    majority_class = Counter(y_train_str).most_common(1)[0][0]
    majority_ct = Counter(y_train_str).most_common(1)[0][1]
    majority_proportion = majority_ct / len(y_train_str)
    y_pred_maj = [majority_class] * len(y_test_str)
    maj_f1 = f1_score(y_test_str, y_pred_maj, average="weighted")
    maj_acc = accuracy_score(y_test_str, y_pred_maj)
    print(f"Always predict '{majority_class}' ({majority_proportion:.1%} of train)")
    print(f"Test F1 (weighted): {maj_f1:.4f}")
    print(f"Test Accuracy: {maj_acc:.4f}")

    # ==============================
    # Summary
    # ==============================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = pd.DataFrame([
        {"model": "Majority Class Baseline", "test_accuracy": round(maj_acc, 4),
         "test_f1_weighted": round(maj_f1, 4)},
        {"model": "Logistic Regression (TF-IDF)", "test_accuracy": round(lr_acc, 4),
         "test_f1_weighted": round(lr_f1, 4)},
        {"model": "XGBoost (TF-IDF)", "test_accuracy": round(xgb_acc, 4),
         "test_f1_weighted": round(xgb_f1, 4)},
    ])
    print(results.to_string(index=False))
    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Feature importance (XGBoost top words)
    print("\n" + "=" * 60)
    print("TOP 20 FEATURES (XGBoost)")
    print("=" * 60)
    tfidf = xgb_pipe.named_steps["tfidf"]
    xgb = xgb_pipe.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    importances = xgb.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]
    print("Most important words overall:")
    for i in top_idx:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")

    # Error analysis: where models disagree with rubric
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    error_df = pd.DataFrame({
        "text": X_test,
        "true_label": y_test_str,
        "lr_pred": y_pred_lr,
        "xgb_pred": y_pred_xgb,
    })
    error_df["lr_correct"] = error_df["true_label"] == error_df["lr_pred"]
    error_df["xgb_correct"] = error_df["true_label"] == error_df["xgb_pred"]
    error_df["both_wrong"] = ~error_df["lr_correct"] & ~error_df["xgb_correct"]
    error_df["lr_only"] = error_df["lr_correct"] != error_df["xgb_correct"]

    print(f"Both models correct: {error_df['lr_correct'].sum()} / {len(error_df)}")
    print(f"LR correct only: {((error_df['lr_correct']) & (~error_df['xgb_correct'])).sum()}")
    print(f"XGB correct only: {((~error_df['lr_correct']) & (error_df['xgb_correct'])).sum()}")
    print(f"Both wrong: {error_df['both_wrong'].sum()}")

    # Show some disagreement cases
    disagreements = error_df[error_df["lr_only"]].head(5)
    if len(disagreements) > 0:
        print("\nSample disagreements (true_label | lr_pred | xgb_pred):")
        for _, row in disagreements.iterrows():
            print(f"  True={row['true_label']} LR={row['lr_pred']} XGB={row['xgb_pred']}")
            print(f"  Text: {row['text'][:100]}...")

    return lr_pipe, xgb_pipe


if __name__ == "__main__":
    train_models()
