# SemEval-2026 Task 10 (PsyCoMark) — Final System

Code for our SemEval-2026 Task 10 (PsyCoMark) systems, described in the paper:

**"UCSC NLP at SemEval-2026 Task 10: Boundary-Aware Span Extraction and RoBERTa Classification for Conspiracy Detection"**
Dom Marhoefer, Aidan Pinero, Glenn Grant-Richards, Milos Suvakovic

Official Codabench results: **7th place in Subtask 1** (span extraction) and **12th place in Subtask 2** (document-level detection).

We implement two models, trained and run independently; outputs are combined only for the Subtask 2 submission:

- **Subtask 1 — conspiracy marker extraction:** span classification with RoBERTa-large. The core model, dataset, and decoding code is `roberta_span_w_summary_v12.py`; the submitted system is the **v25** configuration (`roberta_span_w_summary_v25.py`) with decoding thresholds retuned on validation data (`retune_for_semeval_v25_token05_ws_finer.py`, token IoU 0.5).
- **Subtask 2 — document-level conspiracy detection:** 3-class RoBERTa-large classifier (**v11**), whose submission merges its document predictions with the v25 span markers from Subtask 1.

---

## Official submissions

| Subtask | Submitted file | System |
|---------|---------------|--------|
| 1 | `submission_semeval_token_iou05_v25_ws_finer.zip` | v25 (v12 model + retuned thresholds) |
| 2 | `submission_subtask_2_v11_v25markers.zip` | v11 classifier + v25 markers |

The prediction JSONL/zip artifacts from the official runs are not stored in this repository (file size); everything needed to regenerate them — code, thresholds, and checkpoint metrics — is included. The Subtask 2 submission script takes the Subtask 1 marker predictions via `--markers` (see Run).

## Repository structure

| Path | Description |
|------|-------------|
| `Semeval-Subtask-1/` | Span extraction (RoBERTa-large v25): model, training, decoding, threshold tuning, official submission |
| `Semeval-Subtask-2/` | Document classifier (RoBERTa-large v11): training and submission scripts, official submission |
| `rehydrate_data.py` | Regenerates `train_rehydrated.jsonl` / `test_rehydrated.jsonl` from the task's ID files (see Data) |
| `requirements.txt` | Python dependencies |

See the README in each subtask folder for detailed file roles.

## Setup

```bash
pip install -r requirements.txt
```

## Data

The task data is **not included** in this repository. PsyCoMark distributes Reddit comments by ID; the text must be rehydrated:

1. Obtain the task data (comment IDs and labels) from the official SemEval-2026 Task 10 (PsyCoMark) page.
2. Run `python rehydrate_data.py` to fetch comment text and produce `train_rehydrated.jsonl` and `test_rehydrated.jsonl` at the repo root.

Both subtasks read those two files from the repo root.

## Run

- **Subtask 1:** From `Semeval-Subtask-1/`, run `python roberta_span_w_summary_v25.py` (train) or `python retune_for_semeval_v25_token05_ws_finer.py` (threshold tuning + inference, reproduces the official submission).
- **Subtask 2:** From repo root: `python Semeval-Subtask-2/train_conspiracy_subtask_2_v11.py` (train). For submission: `python Semeval-Subtask-2/submit_subtask_2_v12_v11.py --markers Semeval-Subtask-1/submission_semeval_token_iou05_v25_ws_finer.jsonl` (uses root `test_rehydrated.jsonl` and the v25 marker predictions produced by Subtask 1).

## Model weights

Pretrained weights are **not included** (they exceed GitHub's file size limits and are excluded via `.gitignore`). Both models can be retrained with the provided scripts and the rehydrated data. Tokenizer files and training metrics for the submitted checkpoints are retained under each subtask's `outputs_*/best/` folder for reference.

## License

MIT — see `LICENSE`. Copyright (c) 2026 UC Santa Cruz NLP (Dom Marhoefer, Aidan Pinero, Glenn Grant-Richards, Milos Suvakovic).
