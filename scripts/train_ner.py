"""
Fine-tune a small span-extraction NER on the BIO resume dataset.

A tagger emits only token spans that exist in the input text, so it is
structurally immune to the skill-hallucination the generative LLM shows. This
script trains distilbert-base-uncased as the baseline for that comparison.

Usage:
    python scripts/train_ner.py [--max-steps 100] [--epochs 3]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer,
                          DataCollatorForTokenClassification)

ROOT = Path(__file__).resolve().parent.parent
BASE = "distilbert-base-uncased"
ENTITY_PRIORITY = ["PERSON", "PROJECT", "CERT", "DEGREE", "INSTITUTION",
                   "TITLE", "COMPANY", "SKILL", "LANGUAGE"]
LABELS = ["O"] + [b + "-" + e for e in ENTITY_PRIORITY for b in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({"tokens": d["tokens"], "tags": [LABEL2ID[t] for t in d["tags"]]})
    return rows


def tokenize_batch(examples, tokenizer, max_length=128):
    encoded = tokenizer(examples["tokens"], is_split_into_words=True,
                        truncation=True, max_length=max_length)
    labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = encoded.word_ids(batch_index=i)
        seq, prev = [], None
        for wid in word_ids:
            if wid is None:
                seq.append(-100)
            elif wid != prev:
                seq.append(tags[wid])
            else:
                seq.append(-100)
            prev = wid
        labels.append(seq)
    encoded["labels"] = labels
    return encoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--output-dir", default=str(ROOT / "models" / "ner-v1"))
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE)

    all_train = load(ROOT / "data/processed/ner_tags_train.jsonl")
    all_eval = load(ROOT / "data/processed/ner_tags_val.jsonl")
    ds = Dataset.from_dict({"tokens": [e["tokens"] for e in all_train],
                            "tags": [e["tags"] for e in all_train]})
    de = Dataset.from_dict({"tokens": [e["tokens"] for e in all_eval],
                            "tags": [e["tags"] for e in all_eval]})
    ds = ds.map(lambda ex: tokenize_batch(ex, tokenizer), batched=True,
                remove_columns=["tokens", "tags"])
    de = de.map(lambda ex: tokenize_batch(ex, tokenizer), batched=True,
                remove_columns=["tokens", "tags"])

    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    has_cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs if args.max_steps is None else 0,
        max_steps=args.max_steps,
        logging_steps=50,
        eval_strategy="steps" if args.max_steps else "epoch",
        eval_steps=100 if args.max_steps else None,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        bf16=has_cuda,
        use_cpu=not has_cuda,
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, eval_dataset=de,
                      data_collator=collator)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"NER model saved to {args.output_dir}")


if __name__ == "__main__":
    main()