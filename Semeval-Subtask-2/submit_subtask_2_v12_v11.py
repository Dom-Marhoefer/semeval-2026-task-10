#!/usr/bin/env python3
"""
Build Subtask 2 submission using the v11 conspiracy model (RoBERTa-large, 3-class).

Uses outputs_subtask_2_conspiracy_v11/best by default; writes submission_subtask_2_v12_v11.zip.
Run after: python subtask_2/train_conspiracy_subtask_2_v11.py
"""

import argparse
import json
import os
import zipfile
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

CONSPIRACY_CKPT = os.path.join(_THIS_DIR, "outputs_subtask_2_conspiracy_v11", "best")
MARKERS_JSONL = os.path.join(_ROOT, "submission_semeval_iou095_v12.jsonl")
TEST_JSONL = os.path.join(_ROOT, "test_rehydrated.jsonl")
ZIP_FILE = os.path.join(_THIS_DIR, "submission_subtask_2_v12_v11.zip")
MAX_LENGTH = 512
BATCH_SIZE = 8


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser(description="Build Subtask 2 submission (v11 conspiracy model + v12 markers)")
    parser.add_argument("--input", default=TEST_JSONL, help="Test JSONL")
    parser.add_argument("--markers", default=MARKERS_JSONL, help="V12 markers JSONL")
    parser.add_argument("--conspiracy_ckpt", default=CONSPIRACY_CKPT, help="V11 conspiracy model dir")
    parser.add_argument("--output_zip", default=ZIP_FILE, help="Output zip path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing zip/jsonl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.conspiracy_ckpt):
        raise FileNotFoundError(
            f"Conspiracy checkpoint not found: {args.conspiracy_ckpt}. "
            "Run subtask_2/train_conspiracy_subtask_2_v11.py first."
        )
    if not os.path.isfile(args.markers):
        raise FileNotFoundError(f"Markers file not found: {args.markers}. Run retune_for_semeval_v12.py first.")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Test file not found: {args.input}.")

    label_map_path = os.path.join(args.conspiracy_ckpt, "label_map.json")
    if os.path.isfile(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            idx_to_label = {int(k): v for k, v in json.load(f).items()}
    else:
        idx_to_label = {0: "No", 1: "Yes", 2: "Can't tell"}

    print(f"Loading v11 conspiracy model from {args.conspiracy_ckpt} (3-class)")
    tokenizer = AutoTokenizer.from_pretrained(args.conspiracy_ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.conspiracy_ckpt).to(device)
    model.eval()

    test_items = load_jsonl(args.input)
    markers_by_id = {}
    for rec in load_jsonl(args.markers):
        _id = rec.get("_id")
        if _id is not None:
            markers_by_id[_id] = rec.get("markers") or []

    conspiracy_by_id = {}
    for i in range(0, len(test_items), args.batch_size):
        batch_items = test_items[i : i + args.batch_size]
        texts = []
        ids = []
        for j, it in enumerate(batch_items):
            ids.append(it.get("_id") or it.get("id") or f"ex_{i + j}")
            t = it.get("text") or it.get("rawtext") or it.get("content") or it.get("body") or ""
            texts.append(t)
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        for _id, p in zip(ids, preds):
            conspiracy_by_id[_id] = idx_to_label.get(int(p), "No")

    submission = []
    for idx, it in enumerate(test_items):
        _id = it.get("_id") or it.get("id") or f"ex_{idx}"
        conspiracy = conspiracy_by_id.get(_id, "No")
        markers = markers_by_id.get(_id, [])
        submission.append({"_id": _id, "conspiracy": conspiracy, "markers": markers})

    out_jsonl = args.output_zip.replace(".zip", ".jsonl") if args.output_zip.endswith(".zip") else os.path.join(_THIS_DIR, "submission_subtask_2_v12_v11.jsonl")

    if not args.overwrite and os.path.isfile(args.output_zip):
        print(f"Kept existing {args.output_zip} (use --overwrite to replace)")
        return
    out_dir = os.path.dirname(args.output_zip)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in submission:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(submission)} rows to {out_jsonl}")

    if os.path.exists(args.output_zip):
        os.remove(args.output_zip)
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_jsonl, arcname="submission.jsonl")
    print(f"Created {args.output_zip} (Subtask 2 v11 submission)")


if __name__ == "__main__":
    main()
