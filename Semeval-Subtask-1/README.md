# SemEval-Best — Subtask 1 (span extraction)

Subtask 1 submission: **submission_semeval_token_iou05_v25_ws_finer.zip** (token IoU ≥ 0.5, v25 span model).

## Methods — where to find our approach

| File | What it contains |
|------|------------------|
| **`roberta_span_w_summary_v12.py`** | **Core method:** RoBERTa-large encoder; candidate-span enumeration (token windows); span representation (start/end/mean-pool/width/context → 6d features → MLP → per-role logits); multi-label BCE; IoU-based positive labeling and hard negatives; decoding (per-role threshold → containment NMS → merge); threshold tuning and evaluation loop. |
| **`roberta_span_w_summary_v25.py`** | **Config for this submission:** v25 overrides (OUTPUT_DIR, POS_IOU 0.95 for all five roles, MAX_SPAN_TOKENS=32, MAX_SPANS_PER_EX=160). Imports v12; no new logic. |
| **`retune_for_semeval_v25_token05_ws_finer.py`** | **Inference & submission:** Loads v25 checkpoint; tunes per-role probability thresholds to maximize **macro F1** at **token IoU ≥ 0.5** (whitespace tokenization); runs on validation then test; writes marker JSONL and submission zip. |

Evaluation for the task uses **token IoU ≥ 0.5**; we use **whitespace tokenization** for that metric and tune thresholds on val macro F1.

## Model weights not in this repo

**`outputs_span_multilabel_roberta_decode_v25/best/model.pt`** (~1.4 GB) is not pushed because it exceeds GitHub’s 100 MB limit. To run inference or reproduce the submission you need to either:

- Copy `model.pt` from your training run into `outputs_span_multilabel_roberta_decode_v25/best/`, or  
- Train the v25 model with `python roberta_span_w_summary_v25.py` and use the saved checkpoint.

All other files (scripts, tokenizer, thresholds, data, submission zip) are in the repo. See **SUBMISSION_FILES_FOR_GITHUB_BRANCHES.md** for the full file list.
