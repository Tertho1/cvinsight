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


def load_data(path):
    """Load JSONL and split into train/eval."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    # Use last 200 as eval
    train = examples[:-200]
    eval_data = examples[-200:]
    print(f"Loaded {len(examples)} examples: {len(train)} train, {len(eval_data)} eval")
    return train, eval_data


def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize chat messages for language modeling."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-0.6B")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit training steps for testing")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-eval", action="store_true", help="Skip eval split")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model in BF16 (RTX 5070 Ti supports it natively)
    print(f"Loading model {BASE_MODEL} in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
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
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Load data
    train_data, eval_data = load_data(args.data_path)

    # Convert to Hugging Face Dataset
    train_messages = [ex["messages"] for ex in train_data]
    eval_messages = [ex["messages"] for ex in eval_data]
    train_dataset = Dataset.from_dict({"messages": train_messages})
    eval_dataset = Dataset.from_dict({"messages": eval_messages})

    # Map tokenization
    def _tok(examples):
        return tokenize_function(examples, tokenizer, args.max_length)

    train_dataset = train_dataset.map(_tok, batched=True, remove_columns=["messages"])
    eval_dataset = eval_dataset.map(_tok, batched=True, remove_columns=["messages"])

    # Data collator
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    num_train = len(train_dataset)
    steps_per_epoch = math.ceil(num_train / (args.batch_size * args.grad_accum))

    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=50,
        learning_rate=args.lr,
        bf16=True,
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

    # Train
    print(f"\nStarting training...")
    print(f"  Train examples: {num_train}")
    print(f"  Eval examples: {len(eval_dataset)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {steps_per_epoch * args.num_epochs}")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
