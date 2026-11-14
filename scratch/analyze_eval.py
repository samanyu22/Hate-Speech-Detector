import json

with open("scratch/eval_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

print("=== SUMMARY METRICS ===")
print("Total Samples:", res["total_samples"])
print("Accuracy:", res["accuracy"])
print("Macro F1:", res["macro_f1"])
print("Macro Precision:", res["macro_precision"])
print("Macro Recall:", res["macro_recall"])
print("Confusion Matrix:", res["confusion_matrix"])
print("Per Class:", json.dumps(res["per_class"], indent=2))

print("\n=== TRUE POSITIVES (Hateful Correctly Detected) ===")
for r in res["all_results"]:
    if r["gold_label"] == "hateful" and r["pred_label"] == "hateful":
        print(f"[{r['id']}] {r['text']}")
        print(f"    Prob(Hateful): {r['prob_hateful']:.4f} | Confidence: {r['confidence']:.4f}")

probs_h = [r["prob_hateful"] for r in res["all_results"] if r["gold_label"] == "hateful"]
probs_u = [r["prob_hateful"] for r in res["all_results"] if r["gold_label"] == "unhateful"]

print("\n=== PROBABILITY DISTRIBUTION ===")
print(f"Gold Hateful (N={len(probs_h)}): Min={min(probs_h):.4f}, Mean={sum(probs_h)/len(probs_h):.4f}, Max={max(probs_h):.4f}")
print(f"Gold Unhateful (N={len(probs_u)}): Min={min(probs_u):.4f}, Mean={sum(probs_u)/len(probs_u):.4f}, Max={max(probs_u):.4f}")

print("\n=== SAMPLES OF FALSE NEGATIVES ===")
fn_samples = [r for r in res["all_results"] if r["gold_label"] == "hateful" and r["pred_label"] == "unhateful"]
print(f"Total False Negatives: {len(fn_samples)}")
for r in fn_samples[:15]:
    print(f"[{r['id']}] {r['text']}")
    print(f"    Prob(Hateful): {r['prob_hateful']:.4f} | Pred: {r['pred_label']}")

print("\n=== HIGHEST PROBABILITY FALSE NEGATIVES ===")
fn_sorted = sorted(fn_samples, key=lambda x: x["prob_hateful"], reverse=True)
for r in fn_sorted[:10]:
    print(f"[{r['id']}] Prob: {r['prob_hateful']:.4f} | Text: {r['text']}")

