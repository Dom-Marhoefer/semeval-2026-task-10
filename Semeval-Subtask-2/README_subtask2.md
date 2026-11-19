# Subtask 2 — Conspiracy classification (v11 + v25 markers)

Subtask 2 submission: **submission_subtask_2_v11_v25markers.zip** (v11 3-class classifier + v25 span markers) — 12th place on Codabench. The prediction artifacts themselves are not stored in this repository; regenerate them with the submission script below.

## Methods — where to find our approach

| File | What it contains |
|------|------------------|
| **`train_conspiracy_subtask_2_v11.py`** | **Core method:** RoBERTa-large for sequence classification (3 classes: No / Yes / Can't tell). Stratified train/val split (90/10); label smoothing (0.05); Hugging Face Trainer with early stopping on validation weighted F1. Saves best checkpoint to `outputs_subtask_2_conspiracy_v11/best/`. |
| **`submit_subtask_2_v12_v11.py`** | **Inference & submission:** Loads v11 classifier and a marker JSONL (here: v25 span markers from Subtask 1). Runs document-level classification (max_length=512); merges predicted conspiracy label with markers per document; writes submission JSONL and zip. |

Markers (span predictions) come from Subtask 1; we use **submission_semeval_token_iou05_v25_ws_finer.jsonl** (v25, token IoU 0.5).

## Files in this folder

**Scripts**
- `train_conspiracy_subtask_2_v11.py` — trains v11 RoBERTa-large 3-class classifier
- `submit_subtask_2_v12_v11.py` — builds submission: loads v11 + marker JSONL, writes zip

**Checkpoint (v11)** — `outputs_subtask_2_conspiracy_v11/best/`
- Config, label map, and metrics are included in this repository.
- Tokenizer files and **model.safetensors** (~1.4 GB) are not included (GitHub 100 MB limit). Train with `train_conspiracy_subtask_2_v11.py` or copy them from your own run.

**Data** — not stored in this repository. Regenerate `train_rehydrated.jsonl` and `test_rehydrated.jsonl` at the repo root with `../rehydrate_data.py` (see the top-level README).

To regenerate the submission zip (run from the repo root, after Subtask 1 inference has produced the marker file):  
`python Semeval-Subtask-2/submit_subtask_2_v12_v11.py --markers Semeval-Subtask-1/submission_semeval_token_iou05_v25_ws_finer.jsonl`
