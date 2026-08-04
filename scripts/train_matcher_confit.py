"""
scripts/train_matcher_confit.py
ConFit-style contrastive fine-tune of a small sentence-transformer for resume-JD
matching (ACL 2025, arXiv:2401.16349).

Key ConFit idea: label sparsity is tiny (<0.05% of possible resume-JD pairs), so
train a contrastive (Siamese) encoder that pulls matching pairs together and
pushes random/near-miss pairs apart. We use MultipleNegativesRankingLoss on
human-labeled (resume, matching-JD) pairs from
`cnamuangtoun/resume-job-description-fit` (6,241 train pairs, MIT); the in-batch
JDs of unrelated pairs act as hard negatives automatically.

Train on GPU (RTX 5070 Ti); fall back to CPU. Saves to models/matcher-confit.
Set the app/production embedder to the fine-tuned model via CV_EMBEDDER.

Usage:
    python scripts/train_matcher_confit.py             # full run
    python scripts/train_matcher_confit.py --epochs 1   # quick run
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

POSITIVE_LABELS = {"Good Fit", "Potential Fit"}


def load_pairs(max_train=None, max_test=None):
    from datasets import load_dataset
    tr = load_dataset("cnamuangtoun/resume-job-description-fit", split="train")
    te = load_dataset("cnamuangtoun/resume-job-description-fit", split="test")

    def pairs(ds, cap):
        anchors, positives = [], []
        for row in ds:
            if row["label"] in POSITIVE_LABELS:
                anchors.append(row["resume_text"])
                positives.append(row["job_description_text"])
            if cap and len(anchors) >= cap:
                break
        return anchors, positives

    train_anchors, train_pos = pairs(tr, max_train)
    test_anchors, test_pos = pairs(te, max_test)
    return (train_anchors, train_pos), (test_anchors, test_pos)


def to_dataset(anchors, positives):
    from datasets import Dataset
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def eval_spearman(model, test_anchors, test_positives):
    """Rank correlation between cosine similarity and the ordinal fit label.

    Uses the full held-out "test" split (positive AND no-fit pairs) so the score
    reflects how well the encoder separates matching from non-matching resume-JD
    pairs. Returns (rho, p, n).
    """
    import numpy as np
    from datasets import load_dataset
    from scipy.stats import spearmanr

    te = load_dataset("cnamuangtoun/resume-job-description-fit", split="test")
    rows = []
    for row in te:
        sim = model.encode((row["resume_text"], row["job_description_text"]),
                           normalize_embeddings=True)
        lab = 1.0 if row["label"] in POSITIVE_LABELS else 0.0
        rows.append((float(np.dot(sim[0], sim[1])), lab))
    sims = np.array([r[0] for r in rows])
    labs = np.array([r[1] for r in rows])
    rho, p = spearmanr(sims, labs)
    return rho, p, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--output-dir", default=str(ROOT / "models" / "matcher-confit"))
    ap.add_argument("--warmup", type=float, default=0.1)
    args = ap.parse_args()

    (train_anchors, train_pos), (test_anchors, test_pos) = load_pairs(args.max_train, args.max_test)
    print(f"positive train pairs={len(train_anchors)}  test pairs={len(test_anchors)}")
    train_ds = to_dataset(train_anchors, train_pos)

    from sentence_transformers import SentenceTransformer, trainer, models
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    base = models.Transformer(args.base)
    pool = models.Pooling(base.get_embedding_dimension(),
                          pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[base, pool])

    loss = MultipleNegativesRankingLoss(st_model)

    has_cuda = torch.cuda.is_available()
    steps_per_epoch = max(len(train_anchors) // args.batch_size, 1)
    targs = trainer.SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=int(steps_per_epoch * args.epochs * args.warmup),
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        bf16=has_cuda,
        use_cpu=not has_cuda,
    )

    t = trainer.SentenceTransformerTrainer(
        model=st_model, args=targs, train_dataset=train_ds, loss=loss)
    t.train()
    st_model.save(args.output_dir)

    rho, p, n = eval_spearman(st_model, test_anchors, test_pos)
    print(f"held-out test binary-fit Spearman rho={rho:.4f} (p={p:.2e}, n={n})")
    print(f"Saved -> {args.output_dir}")
    print("Enable in the app:  CV_EMBEDDER=" + args.output_dir)


if __name__ == "__main__":
    main()