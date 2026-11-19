# Subtask 2 only — Document-level conspiracy classification

This folder contains **only** the Subtask 2 (conspiracy document classification) code. No span extraction (Subtask 1) files here.

- **Train:** `python train_conspiracy_subtask_2_v11.py`  
  Uses `train_rehydrated.jsonl` from the repo root (parent dir).
- **Submit:** `python submit_subtask_2_v12_v11.py`  
  Uses v11 classifier + marker JSONL from repo root; writes submission zip.

See **README_subtask2.md** for method details and checkpoint paths.
