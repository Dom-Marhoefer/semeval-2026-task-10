#!/usr/bin/env python3
"""
V11 conspiracy classifier for Subtask 2: v2_more_epochs + stratified split + light label smoothing.

Changes vs v2_more_epochs:
- Stratified train/val split to preserve class ratios.
- Small label smoothing for better generalization.
- Safer TrainingArguments for older transformers; disables wandb by default.

Saves to outputs_subtask_2_conspiracy_v11/best/ — overwrite only when val F1 improves.
"""

import argparse
import json
import os
import inspect
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

# Disable wandb unless explicitly enabled
os.environ.setdefault("WANDB_DISABLED", "true")

SEED = 42
TRAIN_JSONL = os.path.join(_ROOT, "train_rehydrated.jsonl")
OUTPUT_DIR = os.path.join(_THIS_DIR, "outputs_subtask_2_conspiracy_v11")
BEST_DIR = os.path.join(_THIS_DIR, "outputs_subtask_2_conspiracy_v11", "best")
MODEL_NAME = "roberta-large"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 8
EARLY_STOP_PATIENCE = 3
LR = 2e-5
WARMUP_RATIO = 0.1
VAL_RATIO = 0.1
LABEL_SMOOTHING = 0.05

LABEL_MAP = {"No": 0, "Yes": 1, "Can't tell": 2}
IDX_TO_LABEL = {0: "No", 1: "Yes", 2: "Can't tell"}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(jsonl_path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            c = (ex.get("conspiracy") or "").strip()
            if c not in LABEL_MAP:
                continue
            text = ex.get("text") or ex.get("rawtext") or ex.get("content") or ex.get("body") or ""
            if not text:
                continue
            texts.append(text)
            labels.append(LABEL_MAP[c])
    return texts, labels


def stratified_split(
    texts: List[str],
    labels: List[int],
    seed: int = SEED,
    val_ratio: float = VAL_RATIO,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    rng = np.random.RandomState(seed)
    by_label = {}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for _, idxs in by_label.items():
        if not idxs:
            continue
        rng.shuffle(idxs)
        if len(idxs) == 1:
            train_idx.extend(idxs)
            continue
        cut = int((1.0 - val_ratio) * len(idxs))
        cut = max(1, min(cut, len(idxs) - 1))
        train_idx.extend(idxs[:cut])
        val_idx.extend(idxs[cut:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_texts, train_labels, val_texts, val_labels


class ConspiracyDatasetSubtask2V11(Dataset):
    """3-class conspiracy labels (Yes / No / Can't tell)."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.tokenizer = tokenizer
        self.examples = texts
        self.labels = labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx],
        }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1_weighted": f1}


def build_training_args(**kwargs):
    """Filter kwargs to match installed TrainingArguments signature."""
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if "evaluation_strategy" not in sig.parameters:
        for k in ("save_strategy", "load_best_model_at_end", "metric_for_best_model", "greater_is_better", "eval_steps", "save_steps", "save_total_limit"):
            allowed.pop(k, None)
    return TrainingArguments(**allowed)


def main():
    parser = argparse.ArgumentParser(description="Train RoBERTa-large 3-class (v11: stratified split + label smoothing)")
    parser.add_argument("--input", default=TRAIN_JSONL, help="Training JSONL")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Max epochs (default: {EPOCHS})")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE, help=f"Early-stop patience (default: {EARLY_STOP_PATIENCE})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default: {LR})")
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING, help="Label smoothing factor")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    metrics_path = os.path.join(BEST_DIR, "metrics.json")
    prev_f1 = -1.0
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            prev_f1 = float(json.load(f).get("eval_f1_weighted", -1.0))

    texts, labels = load_examples(args.input)
    if len(texts) == 0:
        raise RuntimeError(f"No training examples. Check {args.input} has 'conspiracy' in {{Yes, No, Can't tell}}.")

    train_texts, train_labels, val_texts, val_labels = stratified_split(texts, labels)
    print(
        f"Subtask 2 v11 (3-class) train: {len(train_texts)} examples, val: {len(val_texts)} "
        f"(prev best F1: {prev_f1:.4f})"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds = ConspiracyDatasetSubtask2V11(train_texts, train_labels, tokenizer)
    val_ds = ConspiracyDatasetSubtask2V11(val_texts, val_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    training_args = build_training_args(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=args.label_smoothing,
        report_to="none",
    )

    callbacks = []
    if getattr(training_args, "metric_for_best_model", None):
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    else:
        print("[warn] EarlyStoppingCallback disabled (metric_for_best_model not set in this transformers version).")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    eval_out = trainer.evaluate()
    val_f1 = float(eval_out.get("eval_f1_weighted", 0))
    val_acc = float(eval_out.get("eval_accuracy", 0))
    metrics = {"eval_f1_weighted": val_f1, "eval_accuracy": val_acc}

    if val_f1 > prev_f1:
        trainer.save_model(BEST_DIR)
        tokenizer.save_pretrained(BEST_DIR)
        with open(os.path.join(BEST_DIR, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump(IDX_TO_LABEL, f, indent=2)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"New best: val F1 {val_f1:.4f}, val acc {val_acc:.4f} — saved to {BEST_DIR}")
    else:
        print(f"This run: val F1 {val_f1:.4f}, val acc {val_acc:.4f} — kept previous best (prev F1 {prev_f1:.4f}); nothing overwritten.")


if __name__ == "__main__":
    main()
