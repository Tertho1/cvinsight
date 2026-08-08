"""
Fine-tune a Bangla token-classification NER head (csebuetnlp/banglabert) on the
synthetic Bangla BIO dataset (data/processed/bangla_ner_{train,val,test}.jsonl,
produced by scripts/generate_bangla_ner_dataset.py).

Sequence-tagging is used (rather than the Qwen generative route) so Bangla CV
spans stay faithful to the input text and CPU/GPU inference stays fast (~tens
of ms per resume), matching the English distilbert ner-v1 design.

Usage:
    python scripts/train_bangla_ner.py [--epochs 4] [--batch-size 12]
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
BASE = "csebuetnlp/banglabert"
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


def tokenize_batch(examples, tokenizer, max_length=256):
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
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--output-dir", default=str(ROOT / "models" / "bangla-ner-v1"))
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

    def prep(name):
        rows = load(ROOT / "data/processed" / f"bangla_ner_{name}.jsonl")
        ds = Dataset.from_dict({"tokens": [e["tokens"] for e in rows],
                                "tags": [e["tags"] for e in rows]})
        return ds.map(lambda ex: tokenize_batch(ex, tokenizer, args.max_length),
                      batched=True, remove_columns=["tokens", "tags"])

    ds = prep("train")
    de = prep("val")

    model = AutoModelForTokenClassification.from_pretrained(
            "csebuetnlp/banglabert", num_labels=len(LABELS),
            id2label=ID2LABEL, label2id=LABEL2ID)

    has_cuda = torch.cuda.is_available()
    train_kwargs = {}
    if args.max_steps is not None:
        train_kwargs["max_steps"] = args.max_steps
        train_kwargs["num_train_epochs"] = 0
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        eval_strategy="steps" if args.max_steps is not None else "epoch",
        eval_steps=100 if args.max_steps is not None else None,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        bf16=has_cuda,
        use_cpu=not has_cuda,
        **train_kwargs,
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, eval_dataset=de,
                      data_collator=collator)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Bangla NER model saved to {args.output_dir}")

    test_rows = load(ROOT / "data/processed" / "bangla_ner_test.jsonl")
    print(f"test examples reserved: {len(test_rows)} (eval separately)")


if __name__ == "__main__":
    main()