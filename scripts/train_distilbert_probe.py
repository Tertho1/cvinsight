"""DistilBERT probe for the quality classifier (GPU/CPU aware, fits 8-12GB VRAM).

Trains DistilBERT-base on the rubric-primary corpus raw_text -> {Weak,Average,Strong}
using a stratified-exact 80/20 split (fixed seed). Resumes from the last saved
checkpoint if interrupted. Appends final rubric + human-tier rows to
results/distilbert_results.csv.

Usage: python scripts/train_distilbert_probe.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                          DistilBertForSequenceClassification)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CURATED = ROOT / "data" / "curated"
RESULTS = ROOT / "results"
CKPT = RESULTS / "distilbert_ckpt"
MODEL_NAME = "distilbert-base-uncased"
LABELS = ["Weak", "Average", "Strong"]
SEED = 42
MAX_LEN = 128
BATCH = 32


def main():
    df = pd.read_csv(CURATED / "corpus_primary_v1.csv")
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    x = df["raw_text"].astype(str).tolist()
    y = np.array([LABELS.index(l) for l in df["label"]])

    xtr, xte_, ytr, yte = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    from collections import Counter
    print(f"train {len(xtr)} {Counter(ytr)} | val {len(xte_)} {Counter(yte)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class CVData:
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = list(labels)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tok(self.texts[i], padding="max_length", truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
                    "labels": torch.tensor(self.labels[i], dtype=torch.long)}

    train_ds = CVData(xtr, ytr)
    eval_ds = CVData(xte_, yte)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    args = TrainingArguments(
        output_dir=str(CKPT), num_train_epochs=2, per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=128, learning_rate=2e-5, weight_decay=0.01,
        logging_steps=50, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, report_to=[], seed=SEED, fp16=(device == "cuda"),
    )
    from transformers import Trainer
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=eval_ds)
    ckpts = sorted([p for p in CKPT.rglob("checkpoint-*") if p.is_dir()])
    resume = str(ckpts[-1]) if ckpts else None
    st = time.time()
    trainer.train(resume_from_checkpoint=resume)
    yp = trainer.predict(eval_ds)
    yp_labels = np.argmax(yp.predictions, axis=1)
    acc = accuracy_score(yte, yp_labels)
    f1w = f1_score(yte, yp_labels, average="weighted", zero_division=0)
    f1m = f1_score(yte, yp_labels, average="macro", zero_division=0)
    cm = confusion_matrix(yte, yp_labels, labels=[0, 1, 2])
    row = {"model": "distilbert", "dataset": "rubric80/20", "acc": round(float(acc), 4),
           "f1_weighted": round(float(f1w), 4), "f1_macro": round(float(f1m), 4),
           "time_s": round(time.time() - st, 1)}
    for i, c in enumerate(LABELS):
        row[f"recall_{c}"] = round(float(cm[i, i] / (cm[i].sum() or 1)), 4)
    print("DISTILBERT RESULT:", row)

    rows = [row]
    # preserve eval labels across datasets for human-tier
    eval_map = {}
    for name in ["ats", "netsol"]:
        dt = pd.read_csv(CURATED / f"corpus_{name}_v1.csv")
        dt["label"] = dt["label"].astype(str).str.strip()
        dt = dt[dt["label"].isin(LABELS)]
        xd = dt["raw_text"].astype(str).tolist()
        yd = np.array([LABELS.index(l) for l in dt["label"]])
        pred = trainer.predict(CVData(xd, yd))
        ypd = np.argmax(pred.predictions, axis=1)
        accd = accuracy_score(yd, ypd)
        f1wd = f1_score(yd, ypd, average="weighted", zero_division=0)
        f1md = f1_score(yd, ypd, average="macro", zero_division=0)
        cmv = confusion_matrix(yd, ypd, labels=[0, 1, 2])
        r2 = {"model": "distilbert", "dataset": name, "acc": round(float(accd), 4),
              "f1_weighted": round(float(f1wd), 4), "f1_macro": round(float(f1md), 4)}
        for i, c in enumerate(LABELS):
            r2[f"recall_{c}"] = round(float(cmv[i, i] / (cmv[i].sum() or 1)), 4)
        rows.append(r2)
        print("DISTILBERT VAL", name, r2)

    out = RESULTS / "distilbert_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("PROBE DONE ->", out)


if __name__ == "__main__":
    main()