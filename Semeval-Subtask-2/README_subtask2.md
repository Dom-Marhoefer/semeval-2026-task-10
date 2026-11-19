# Subtask 2 — Conspiracy classification (v11 + v25 markers)

Subtask 2 submission: **submission_subtask_2_v11_v25markers.zip** (v11 3-class classifier + v25 span markers).

## Methods — where to find our approach

| File | What it contains |
|------|------------------|
| **`train_conspiracy_subtask_2_v11.py`** | **Core method:** RoBERTa-large for sequence classification (3 classes: No / Yes / Can't tell). Stratified train/val split (90/10); label smoothing (0.05); Hugging Face Trainer with early stopping on validation weighted F1. Saves best checkpoint to `outputs_subtask_2_conspiracy_v11/best/`. |
| **`submit_subtask_2_v12_v11.py`** | **Inference & submission:** Loads v11 classifier and a marker JSONL (here: v25 span markers from Subtask 1). Runs document-level classification (max_length=512); merges predicted conspiracy label with markers per document; writes submission JSONL and zip. |

Markers (span predictions) come from Subtask 1; we use **submission_semeval_token_iou05_v25_ws_finer.jsonl** (v25, token IoU 0.5).

## Files on this branch

**Scripts**
- `train_conspiracy_subtask_2_v11.py` — trains v11 RoBERTa-large 3-class classifier
- `submit_subtask_2_v12_v11.py` — builds submission: loads v11 + marker JSONL, writes zip

**Checkpoint (v11)** — `outputs_subtask_2_conspiracy_v11/best/`
- Config, tokenizer, label_map, metrics (in repo)
- **model.safetensors** (~1.4 GB) not in repo (GitHub 100 MB limit). Train with `train_conspiracy_subtask_2_v11.py` or copy from your run.

**Submission**
- `submission_subtask_2_v11_v25markers.jsonl`
- `submission_subtask_2_v11_v25markers.zip`

**Data (repo root)**  
- `train_rehydrated.jsonl`, `test_rehydrated.jsonl` (in repo root)  
- Marker JSONL (e.g. `submission_semeval_token_iou05_v25_ws_finer.jsonl` from Subtask 1) in repo root.

To regenerate the zip (run from `Semeval-Subtask-2/`):  
`python submit_subtask_2_v12_v11.py --markers ../submission_semeval_token_iou05_v25_ws_finer.jsonl --output_zip submission_subtask_2_v12_v11.zip`
