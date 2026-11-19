"""
roberta_span_w_summary_v25.py

v25 = v24 with stricter positives for all roles:
- MAX_SPAN_TOKENS = 32
- MAX_SPANS_PER_EX = 160
- OUTPUT_DIR updated to v25
"""

import roberta_span_w_summary_v12 as v12

# -------------------------
# v25 overrides (targeted changes only)
# -------------------------
v12.OUTPUT_DIR = "./outputs_span_multilabel_roberta_decode_v25"
v12.MAX_SPAN_TOKENS = 32
v12.MAX_SPANS_PER_EX = 160

# Stricter positives for all roles
v12.POS_IOU_THR_BY_ROLE = {
    "Actor": 0.95,
    "Victim": 0.95,
    "Action": 0.95,
    "Effect": 0.95,
    "Evidence": 0.95,
}
v12.POS_IOU_THR = 0.75  # fallback (unused for roles above)


# -------------------------
# Summary printing (override header to v25)
# -------------------------
def print_training_summary(config, best_metrics, pos_weight, total_time_s):
    print("\n" + "=" * 90)
    print("TRAINING SUMMARY - ALL MARKERS (SPAN MULTI-LABEL + DECODING) [V25 / RoBERTa-large]")
    print("=" * 90)

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nBest Validation Metrics (by decoded_micro_f1):")
    print(f"  decoded_micro_f1: {best_metrics.get('decoded_micro_f1', 0.0):.4f}")
    print(f"  decoded_micro_p:  {best_metrics.get('decoded_micro_p', 0.0):.4f}")
    print(f"  decoded_micro_r:  {best_metrics.get('decoded_micro_r', 0.0):.4f}")
    print(f"  cand_micro_f1:    {best_metrics.get('cand_micro_f1', 0.0):.4f}")
    print(f"  n_examples_eval:  {best_metrics.get('n_examples_eval', 0)}")
    print(f"  n_spans_eval:     {best_metrics.get('n_spans_eval', 0)}")

    print("\nThresholds (decoded):")
    th = best_metrics.get("thresholds", {})
    for r in v12.ROLES:
        print(f"  {r:<8} {th.get(r, v12.DEFAULT_THRESH):.2f}")

    print("\nRole-wise metrics:")
    print(f"{'Role':<10} {'CandF1':<10} {'DecP':<10} {'DecR':<10} {'DecF1':<10} {'PosWeight':<10}")
    print("-" * 70)

    pw = pos_weight.detach().cpu().numpy().tolist()
    cand_role = best_metrics.get("cand_role_f1", {})
    dec_role = best_metrics.get("decoded_role_prf", {})

    for i, r in enumerate(v12.ROLES):
        cf1 = cand_role.get(r, 0.0)
        dp = dec_role.get(r, {}).get("p", 0.0)
        dr = dec_role.get(r, {}).get("r", 0.0)
        df1 = dec_role.get(r, {}).get("f1", 0.0)
        print(f"{r:<10} {cf1:<10.4f} {dp:<10.4f} {dr:<10.4f} {df1:<10.4f} {pw[i]:<10.2f}")

    print("\nRuntime:")
    print(f"  total_seconds: {total_time_s:.2f}")
    print("=" * 90)
    print(f"Training completed. Models saved to: {v12.OUTPUT_DIR}")
    print("=" * 90)


v12.print_training_summary = print_training_summary


if __name__ == "__main__":
    v12.main()
