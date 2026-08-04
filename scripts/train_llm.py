"""
LoRA fine-tune Qwen3-0.6B on CV extraction training data.

Usage:
    python scripts/train_llm.py                     # Full training
    python scripts/train_llm.py --max-steps 100     # Quick test
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# PEFT <-> transformers lazy-import shim: transformers 5.x dropped the top-level
# Bloom symbol that peft loads eagerly. Re-register it before peft imports.
import importlib as _importlib
_tf = _importlib.import_module("transformers")
try:
    _bloom = _tf.models.bloom.modeling_bloom.BloomPreTrainedModel
    _tf.BloomPreTrainedModel = _bloom
except Exception:
    pass

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)

BASE_MODEL = "Qwen/Qwen3-0.6B"
DATA_PATH = "data/processed/training_data.jsonl"
TRAIN_PATH = "data/processed/curated_curated_train.jsonl"
EVAL_PATH = "data/processed/curated_curated_val.jsonl"
OUTPUT_DIR = "models/qwen3-0.6b-cv-lora"

# Qwen3 uses <|im_start|>/<|im_end|> chat template — we rely on apply_chat_template
SYSTEM_PROMPT = (
    "You are an expert resume parser. "
    "Extract structured information from resumes and return ONLY valid JSON. "
    "Do not include explanations or extra text."
)


def format_chat(example):
    """Convert chat messages to tokenized IDs using the model's chat template."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Resume:\n{example['resume_text']}"},
            {"role": "assistant", "content": json.dumps(example["extracted_json"], ensure_ascii=False, indent=2)},
        ]
    }


def load_data(paths):
    """Load chat-format JSONL from one or more paths into a flat example list."""
    examples = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    print(f"Loaded {len(examples)} chat examples: {', '.join(paths)}")
    return examples


def load_split(args):
    """Pick the train/eval JSONL files (curated split by default)."""
    if args.train_path or args.data_path:
        # Legacy single-file mode: keep last N as eval
        path = args.train_path or args.data_path
        all_ex = load_data([path])
        eval_n = args.eval_size or 200
        return all_ex[:-eval_n], all_ex[-eval_n:]
    return load_data([args.train_path or TRAIN_PATH]), load_data([EVAL_PATH])


def _newest_checkpoint(output_dir):
    """Return the most recent 'checkpoint-*' subdir for trainer resume."""
    out = Path(output_dir)
    if not out.exists():
        return None
    steps = sorted(
        [int(p.name.rsplit("-", 1)[1]) for p in out.glob("checkpoint-*")
         if p.is_dir() and p.name.rsplit("-", 1)[1].isdigit()]
    )
    return str(out / f"checkpoint-{steps[-1]}") if steps else None


def tokenize_masked(examples, tokenizer, max_length=2048):
    """Tokenize chat messages but only compute loss on the assistant answer.

    The user prompt (system + resume) is tokenized separately; any token before
    the assistant's JSON is masked to -100 so the model only learns to produce
    structured output, not to reproduce the resume.
    """
    prompt_texts, full_texts = [], []
    for messages in examples["messages"]:
        prompt_texts.append(
            tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
        )
        full_texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )

    fp = tokenizer(prompt_texts, truncation=True, max_length=max_length, padding=False)
    ff = tokenizer(full_texts, truncation=True, max_length=max_length, padding=False)

    input_ids, labels = [], []
    for pid, fid in zip(fp["input_ids"], ff["input_ids"]):
        mask_len = min(len(pid), len(fid))
        labels.append([-100] * mask_len + fid[mask_len:])
        input_ids.append(fid)
    return {"input_ids": input_ids, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-0.6B")
    parser.add_argument("--train-path", default=None, help="Curated train JSONL (default: curated split)")
    parser.add_argument("--eval-path", default=None, help="Curated eval JSONL")
    parser.add_argument("--data-path", default=None, help="Legacy: single-file mode (last N = eval)")
    parser.add_argument("--eval-size", type=int, default=200, help="Legacy single-file eval size")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit training steps for testing")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--rslora", action="store_true", default=True, help="Use rsLoRA scaling")
    parser.add_argument("--neftune", type=float, default=0, help="NEFTune noise magnitude (e.g. 5)")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-eval", action="store_true", help="Skip eval split")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from the newest checkpoint in --output-dir (for power-loss)")
    parser.add_argument("--resume-from", default=None,
                        help="Explicit checkpoint dir to resume from (e.g. a preserved run)")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model in BF16 (RTX 5070 Ti supports it natively)
    # Load model — float32 on CPU, BF16 on GPU
    print(f"Loading model {BASE_MODEL} ...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        use_rslora=args.rslora,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Load data (separate curated split, or legacy single-file)
    eval_data = None
    if args.no_eval:
        train_data = load_data([args.train_path or (args.data_path or TRAIN_PATH)])
    else:
        if args.train_path or args.data_path or args.eval_path:
            train_data, eval_data = load_function(args)
        else:
            train_data = load_data([TRAIN_PATH])
            eval_data = load_data([EVAL_PATH])

    # Convert to Hugging Face Dataset
    train_messages = [ex["messages"] for ex in train_data]
    train_dataset = Dataset.from_dict({"messages": train_messages})
    if eval_data:
        eval_messages = [ex["messages"] for ex in eval_data]
        eval_dataset = Dataset.from_dict({"messages": eval_messages})

    # Map tokenization (labels masked to assistant JSON only)
    def _tok(examples):
        return tokenize_masked(examples, tokenizer, args.max_length)

    train_dataset = train_dataset.map(_tok, batched=True, remove_columns=["messages"])
    if eval_data:
        eval_dataset = eval_dataset.map(_tok, batched=True, remove_columns=["messages"])

    # Data collator
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    num_train = len(train_dataset)
    steps_per_epoch = math.ceil(num_train / (args.batch_size * args.grad_accum))
    has_cuda = torch.cuda.is_available()

    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=50,
        learning_rate=args.lr,
        bf16=has_cuda,
        use_cpu=not has_cuda,
        logging_steps=10,
        save_steps=steps_per_epoch,
        save_total_limit=2,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=steps_per_epoch if eval_data else None,
        load_best_model_at_end=True if eval_data else False,
        metric_for_best_model="eval_loss" if eval_data else None,
        report_to="none",
        dataloader_num_workers=0,
    )
    if args.neftune > 0:
        kwargs["neftune_noise_alpha"] = args.neftune
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    training_args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_data else None,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # Train (with checkpoint resume for power-loss recovery)
    print(f"\nStarting training...")
    print(f"  Train examples: {num_train}")
    print(f"  Eval examples: {len(eval_dataset) if eval_data else 0}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {steps_per_epoch * args.num_epochs}")

    resume_from = None
    if args.resume_from:
        resume_from = args.resume_from
        print(f"Resuming from explicit checkpoint: {resume_from}")
    elif args.resume:
        resume_from = _newest_checkpoint(args.output_dir)
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")
        else:
            print("--resume given but no checkpoint found; starting fresh.")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - t0

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
