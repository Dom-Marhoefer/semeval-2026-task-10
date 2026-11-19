#!/usr/bin/env python3
"""
Re-tune v25 thresholds using token-based IoU >= 0.5 (whitespace tokens) and generate submission.
Objective: maximize MACRO F1 with a finer, local grid around previous best thresholds.
Uses v25 weights (no retraining).
"""

import argparse
import json
import os
import re
import zipfile
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader

import roberta_span_w_summary_v25 as v25

m = v25.v12

SEMEVAL_TOKEN_IOU_THR = 0.50
CKPT_DIR = "outputs_span_multilabel_roberta_decode_v25/best"
BASE_THRESHOLDS_FILE = "best_thresholds_semeval_token_iou05_v25_ws.json"
THRESHOLDS_FILE = "best_thresholds_semeval_token_iou05_v25_ws_finer.json"
OUTPUT_JSONL = "submission_semeval_token_iou05_v25_ws_finer.jsonl"
ZIP_FILE = "submission_semeval_token_iou05_v25_ws_finer.zip"
LOG_FILE = "retune_results_v25_token05_ws_finer.log"


def whitespace_token_spans(text: str):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def char_span_to_token_set(span_char, token_spans):
    s_char, e_char = span_char
    if e_char <= s_char:
        return set()
    tokens = set()
    for i, (s_tok, e_tok) in enumerate(token_spans):
        if e_tok <= s_tok:
            continue
        if max(s_char, s_tok) < min(e_char, e_tok):
            tokens.add(i)
    return tokens


def spans_char_to_token_sets(spans_char, token_spans):
    token_sets = []
    for span in spans_char:
        ts = char_span_to_token_set(span, token_spans)
        if ts:
            token_sets.append(ts)
    return token_sets


def token_iou(tokens1, tokens2):
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    inter = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return inter / union if union > 0 else 0.0


def match_pred_to_gold(pred_sets, gold_sets, match_iou_thr):
    if not pred_sets and not gold_sets:
        return (0, 0, 0)
    if not pred_sets:
        return (0, 0, len(gold_sets))
    if not gold_sets:
        return (0, len(pred_sets), 0)
    gold_used = [False] * len(gold_sets)
    tp = fp = 0
    for p in pred_sets:
        best_j, best_iou = None, 0.0
        for j, g in enumerate(gold_sets):
            if gold_used[j]:
                continue
            v = token_iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j is not None and best_iou >= match_iou_thr:
            tp += 1
            gold_used[best_j] = True
        else:
            fp += 1
    return (tp, fp, gold_used.count(False))


def prf_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return (p, r, f)


@torch.no_grad()
def tune_thresholds_strict(model, loader, device, match_iou_thr, base_thresholds):
    thresholds = dict(base_thresholds)
    model.eval()
    cached = []
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        spans = batch.spans.to(device)
        logits = model(input_ids, attention_mask, spans)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        gold_list = batch.gold_char_by_role
        token_spans_list = [whitespace_token_spans(t) for t in batch.text]
        gold_token = []
        for b in range(probs.shape[0]):
            per_role = {}
            token_spans = token_spans_list[b]
            for role in m.ROLES:
                per_role[role] = spans_char_to_token_sets(gold_list[b].get(role, []), token_spans)
            gold_token.append(per_role)
        cached.append({
            "probs": probs,
            "spans": batch.spans.detach().cpu().numpy(),
            "offsets": batch.offsets.detach().cpu().numpy(),
            "mask": batch.span_mask.detach().cpu().numpy(),
            "gold_token": gold_token,
            "token_spans": token_spans_list,
        })

    def macro_f1_for_thresholds(temp_thr: dict) -> float:
        role_counts = {r: {"tp": 0, "fp": 0, "fn": 0} for r in m.ROLES}
        for pack in cached:
            B = pack["probs"].shape[0]
            for b in range(B):
                decoded = m.decode_example(
                    probs_ex=pack["probs"][b],
                    spans_tok_ex=pack["spans"][b],
                    offsets_ex=pack["offsets"][b],
                    span_mask_ex=pack["mask"][b],
                    thresholds=temp_thr,
                )
                token_spans = pack["token_spans"][b]
                for role in m.ROLES:
                    pred_sets = spans_char_to_token_sets(decoded[role], token_spans)
                    gold_sets = pack["gold_token"][b].get(role, [])
                    tp, fp, fn = match_pred_to_gold(pred_sets, gold_sets, match_iou_thr)
                    role_counts[role]["tp"] += tp
                    role_counts[role]["fp"] += fp
                    role_counts[role]["fn"] += fn
        role_f1s = [
            prf_from_counts(role_counts[r]["tp"], role_counts[r]["fp"], role_counts[r]["fn"])[2]
            for r in m.ROLES
        ]
        return float(sum(role_f1s) / len(role_f1s)) if role_f1s else 0.0

    print("\nTuning thresholds with token IoU >= {:.2f} (v25, macro F1, finer grid)...".format(match_iou_thr))
    for role in m.ROLES:
        base = thresholds[role]
        grid_start = max(0.05, round(base - 0.06, 2))
        grid_end = min(0.95, round(base + 0.06, 2))
        grid = [round(x, 2) for x in np.arange(grid_start, grid_end + 1e-9, 0.01)]
        best_thr = thresholds[role]
        best_f1 = -1.0
        for thr in grid:
            temp_thr = dict(thresholds)
            temp_thr[role] = thr
            macro_f1 = macro_f1_for_thresholds(temp_thr)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_thr = thr
        thresholds[role] = best_thr
        print(f"  {role}: best_thr={best_thr:.2f} macro_f1={best_f1:.4f}")
    return thresholds


@torch.no_grad()
def evaluate_strict(model, loader, device, thresholds, match_iou_thr):
    model.eval()
    role_counts = {r: {"tp": 0, "fp": 0, "fn": 0} for r in m.ROLES}
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        spans = batch.spans.to(device)
        logits = model(input_ids, attention_mask, spans)
        probs_np = torch.sigmoid(logits).detach().cpu().numpy()
        spans_np = batch.spans.detach().cpu().numpy()
        offsets_np = batch.offsets.detach().cpu().numpy()
        mask_np = batch.span_mask.detach().cpu().numpy()
        gold_list = batch.gold_char_by_role
        token_spans_list = [whitespace_token_spans(t) for t in batch.text]
        B = probs_np.shape[0]
        for b in range(B):
            decoded = m.decode_example(
                probs_ex=probs_np[b],
                spans_tok_ex=spans_np[b],
                offsets_ex=offsets_np[b],
                span_mask_ex=mask_np[b],
                thresholds=thresholds,
            )
            token_spans = token_spans_list[b]
            for role in m.ROLES:
                pred_sets = spans_char_to_token_sets(decoded[role], token_spans)
                gold_sets = spans_char_to_token_sets(gold_list[b].get(role, []), token_spans)
                tp, fp, fn = match_pred_to_gold(pred_sets, gold_sets, match_iou_thr)
                role_counts[role]["tp"] += tp
                role_counts[role]["fp"] += fp
                role_counts[role]["fn"] += fn
    micro_tp = sum(role_counts[r]["tp"] for r in m.ROLES)
    micro_fp = sum(role_counts[r]["fp"] for r in m.ROLES)
    micro_fn = sum(role_counts[r]["fn"] for r in m.ROLES)
    decoded_p, decoded_r, decoded_f1 = prf_from_counts(micro_tp, micro_fp, micro_fn)
    decoded_role_prf = {
        r: dict(zip(("p", "r", "f1"), prf_from_counts(role_counts[r]["tp"], role_counts[r]["fp"], role_counts[r]["fn"])))
        for r in m.ROLES
    }
    macro_f1 = float(sum(decoded_role_prf[r]["f1"] for r in m.ROLES) / len(m.ROLES))
    return {
        "decoded_micro_f1": float(decoded_f1),
        "decoded_micro_p": float(decoded_p),
        "decoded_micro_r": float(decoded_r),
        "decoded_macro_f1": macro_f1,
        "decoded_role_prf": decoded_role_prf,
    }


def predict_conspiracy_from_markers(markers):
    return "Yes" if len(markers) > 0 else "No"


def _load_tokenizer():
    tok_source = CKPT_DIR
    tok_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
    if not (os.path.isdir(CKPT_DIR) and any(os.path.isfile(os.path.join(CKPT_DIR, f)) for f in tok_files)):
        tok_source = m.MODEL_NAME
    local_only = bool(os.environ.get("TRANSFORMERS_OFFLINE"))
    return m.AutoTokenizer.from_pretrained(
        tok_source, use_fast=True, add_prefix_space=True, local_files_only=local_only
    )


def _log_result(macro_f1, micro_f1, thresholds):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts}\tv25_token05_ws_finer\tmacro_f1={macro_f1:.4f}\tmicro_f1={micro_f1:.4f}\tthr={thresholds}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def _make_val_loader(tokenizer):
    base_ds = m.SpanMultiLabelDataset(m.TRAIN_JSONL, tokenizer, split="val")

    class ValWithText(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
            self.examples = ds.examples

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            item["text"] = self.examples[idx]["text"]
            return item

    def collate_with_text(batch_list):
        collate_fn = m.make_collate_fn(tokenizer)
        batch = collate_fn(batch_list)
        batch.text = [x["text"] for x in batch_list]
        return batch

    val_ds = ValWithText(base_ds)
    val_loader = DataLoader(val_ds, batch_size=m.BATCH_SIZE, shuffle=False, collate_fn=collate_with_text)
    return val_loader


def _load_base_thresholds():
    if os.path.isfile(BASE_THRESHOLDS_FILE):
        with open(BASE_THRESHOLDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"Actor": 0.30, "Action": 0.25, "Effect": 0.25, "Evidence": 0.25, "Victim": 0.25}


def main():
    ap = argparse.ArgumentParser(description="Re-tune v25 thresholds for token IoU 0.5 (whitespace) and generate submission")
    ap.add_argument("--input", default=os.path.join(m._REPO_ROOT, "test_rehydrated.jsonl"), help="Test JSONL (default: repo root test_rehydrated.jsonl)")
    args = ap.parse_args()

    print("=" * 80)
    print("Re-tuning v25 (RoBERTa-large) for SemEval (token IoU >= 0.5, whitespace, macro F1, finer grid)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = _load_tokenizer()
    model = m.SpanMultiLabelModel(m.MODEL_NAME, num_roles=len(m.ROLES)).to(device)
    state = torch.load(f"{CKPT_DIR}/model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded v25 model from {CKPT_DIR}")

    val_loader = _make_val_loader(tokenizer)
    print(f"Validation set: {len(val_loader.dataset)} examples")

    base_thresholds = _load_base_thresholds()
    print(f"Starting from base thresholds: {base_thresholds}")
    new_thresholds = tune_thresholds_strict(model, val_loader, device, SEMEVAL_TOKEN_IOU_THR, base_thresholds)

    print("\n" + "=" * 80)
    print("Evaluation with Re-tuned Thresholds (token IoU >= {:.2f})".format(SEMEVAL_TOKEN_IOU_THR))
    print("=" * 80)
    metrics = evaluate_strict(model, val_loader, device, new_thresholds, SEMEVAL_TOKEN_IOU_THR)
    print(f"\nMicro: F1={metrics['decoded_micro_f1']:.4f} P={metrics['decoded_micro_p']:.4f} R={metrics['decoded_micro_r']:.4f}")
    print(f"Macro: F1={metrics['decoded_macro_f1']:.4f}")
    print(f"\nPer-role F1:")
    for r in m.ROLES:
        print(f"  {r}: {metrics['decoded_role_prf'][r]['f1']:.4f}")

    with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(new_thresholds, f, indent=2)
    print(f"\nSaved thresholds to: {THRESHOLDS_FILE}")
    _log_result(metrics["decoded_macro_f1"], metrics["decoded_micro_f1"], new_thresholds)

    if not os.path.isfile(args.input):
        raise FileNotFoundError(
            f"Test file not found: {args.input}. Run rehydrate_data.py on test_redacted.jsonl to create test_rehydrated.jsonl."
        )
    print("\n" + "=" * 80)
    print("Generating test predictions (v25)")
    print(f"Test input: {args.input}")
    print("=" * 80)

    class InferDataset(torch.utils.data.Dataset):
        def __init__(self, jsonl_path, tokenizer):
            self.tokenizer = tokenizer
            self.examples = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            text = ex.get("text") or ex.get("rawtext") or ex.get("content") or ex.get("body") or ""
            _id = ex.get("_id") or ex.get("id") or f"ex_{idx}"
            enc = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512,
                                 add_special_tokens=True, return_attention_mask=True)
            offsets = enc["offset_mapping"]
            valid = [i for i, (s, e) in enumerate(offsets) if e > s]
            spans = []
            for i_pos, i in enumerate(valid):
                for j_pos in range(i_pos, min(i_pos + m.MAX_SPAN_TOKENS, len(valid))):
                    j = valid[j_pos]
                    spans.append((i, j))
            return {
                "id": _id,
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "offsets": torch.tensor(offsets, dtype=torch.long),
                "spans": torch.tensor(spans, dtype=torch.long),
            }

    def collate_infer(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        max_spans = max(len(b["spans"]) for b in batch)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        input_ids, attention_mask, offsets, spans, span_mask, ids = [], [], [], [], [], []
        for b in batch:
            pad_t = max_len - len(b["input_ids"])
            pad_s = max_spans - len(b["spans"])
            input_ids.append(torch.cat([b["input_ids"], torch.full((pad_t,), pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_t, dtype=torch.long)]))
            offsets.append(torch.cat([b["offsets"], torch.zeros((pad_t, 2), dtype=torch.long)]))
            spans.append(torch.cat([b["spans"], torch.full((pad_s, 2), -1, dtype=torch.long)]))
            span_mask.append(torch.cat([torch.ones(len(b["spans"]), dtype=torch.bool), torch.zeros(pad_s, dtype=torch.bool)]))
            ids.append(b["id"])
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "offsets": torch.stack(offsets),
            "spans": torch.stack(spans),
            "span_mask": torch.stack(span_mask),
            "ids": ids,
        }

    test_ds = InferDataset(args.input, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=m.BATCH_SIZE, shuffle=False, collate_fn=collate_infer)

    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["spans"].to(device))
            probs = torch.sigmoid(logits).cpu().float().numpy()
            for b in range(probs.shape[0]):
                decoded = m.decode_example(
                    probs_ex=probs[b],
                    spans_tok_ex=batch["spans"].cpu().numpy()[b],
                    offsets_ex=batch["offsets"].cpu().numpy()[b],
                    span_mask_ex=batch["span_mask"].cpu().numpy()[b],
                    thresholds=new_thresholds,
                )
                markers = [{"startIndex": int(s), "endIndex": int(e), "type": role}
                           for role in m.ROLES for (s, e) in decoded[role]]
                all_predictions.append({
                    "_id": batch["ids"][b],
                    "conspiracy": predict_conspiracy_from_markers(markers),
                    "markers": markers,
                })

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in all_predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_predictions)} predictions to: {OUTPUT_JSONL}")

    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_JSONL, arcname="submission.jsonl")
    print(f"Created submission zip: {ZIP_FILE}")

    print("\n" + "=" * 80)
    print("Summary (v25 / RoBERTa-large)")
    print("=" * 80)
    for role, thr in new_thresholds.items():
        print(f"  {role}: {thr:.2f}")
    print(f"\nValidation Micro F1 @ token IoU 0.5: {metrics['decoded_micro_f1']:.4f}")
    print(f"Validation Macro F1 @ token IoU 0.5: {metrics['decoded_macro_f1']:.4f}")
    print(f"Test input used: {args.input}")
    print(f"Files: {THRESHOLDS_FILE}, {OUTPUT_JSONL}, {ZIP_FILE}")
    print("Ready for submission. Upload {} to SemEval/CodaLab.".format(ZIP_FILE))


if __name__ == "__main__":
    main()
