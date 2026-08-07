"""
Bangla CV section classifier — train on the Onneshon resume dataset.

Onneshon (Mendeley 10.17632/4md7bx6fd7.1) is a 1,739-segment Bangla resume corpus
labelled at the section level: Objective / Experience / Skill / Education.

This trains a lightweight char n-gram TF-IDF + Logistic Regression classifier
(the problem is easy; see docs/research_bangla_cv_support.md — no BanglaBERT needed)
and saves it to models/bangla_section_classifier.pkl. The classifier tags an
arbitrary Bangla resume segment with one of those four section labels, providing a
native Bangla sectioning signal that slots beside src/parser/section_splitter.py.

Usage:
    python scripts/train_bangla_section_classifier.py

Metrics are written to models/bangla_section_eval.json
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

DATA_FILE = "data/raw/onneshon_raw.csv"
MODEL_PATH = "models/bangla_section_classifier.pkl"
EVAL_PATH = "models/bangla_section_eval.json"
SEED = 42


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, accuracy_score, f1_score)


def load_and_clean():
    df = pd.read_csv(DATA_FILE)
    before = len(df)
    df = df.drop_duplicates(subset="text")
    print(f"Loaded {before} segments; {before - len(df)} exact-dup removed -> {len(df)}")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def main():
    df = load_and_clean()
    X = df["text"].to_list()
    y = df["label"].to_numpy()

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2,
                          sublinear_tf=True)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=SEED)

    # Cross-validated evaluation (stratified; duplicates already removed so no leakage)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr_i, te_i in skf.split(X, y):
        x_tr = vec.fit_transform([X[i] for i in tr_i])
        x_te = vec.transform([X[i] for i in te_i])
        c = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                               random_state=SEED).fit(x_tr, y[tr_i])
        p = c.predict(x_te)
        accs.append(accuracy_score(y[te_i], p))
        f1s.append(f1_score(y[te_i], p, average="macro"))
    print("\n5-fold CV micro/macro summary:")
    print("  accuracy mean:", round(np.mean(accs), 4), accs)
    print("  macro-F1 mean:", round(np.mean(f1s), 4))

    # Retrain on 100% for the artifact + report a held-out report
    Xtr, ytr = X, y
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        Xtr, ytr, test_size=0.25, random_state=SEED, stratify=ytr)
    x_tr = vec.fit_transform(X_train)
    x_te = vec.transform(X_test)
    clf.fit(x_tr, y_train)
    pred = clf.predict(x_te)

    print("\nHeld-out classification report (25% split):")
    print(classification_report(y_test, pred, digits=3))

    report = classification_report(y_test, pred, output_dict=True)

    pipe = {"vectorizer": vec, "classifier": clf,
            "classes": list(clf.classes_)}
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

    eval_out = {
        "dataset": DATA_FILE,
        "n_segments": int(len(df)),
        "after_dedup": int(len(df)),
        "cv5_accuracy_mean": round(float(np.mean(accs)), 4),
        "cv5_accuracy_all": [round(float(a), 4) for a in accs],
        "cv5_macro_f1_mean": round(float(np.mean(f1s)), 4),
        "heldout_accuracy": round(accuracy_score(y_test, pred), 4),
        "heldout_weighted_f1": round(f1_score(y_test, pred, average="weighted"), 4),
        "baseline_majority": round(float(max(Counter(y_test).values()) / len(y_test)), 4),
        "classes": sorted(set(y)),
        "report": report,
    }
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, ensure_ascii=False, indent=2)
    print(f"Saved eval -> {EVAL_PATH}")
    return pipe


if __name__ == "__main__":
    main()