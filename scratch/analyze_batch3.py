import json

with open("scratch/eval_batch3_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

print(f"Total Mismatches: {len(res['mismatches'])}")

fps = [m for m in res["mismatches"] if m["gold_label"] == "unhateful" and m["pred_label"] == "hateful"]
fns = [m for m in res["mismatches"] if m["gold_label"] == "hateful" and m["pred_label"] == "unhateful"]

print(f"\n=== FALSE POSITIVES ({len(fps)}) [Gold: unhateful -> Pred: hateful] ===")
for m in fps:
    print(f"[{m['id']}] Prob: {m['prob_hateful']:.4f} | Conf: {m['confidence']:.4f}")
    print(f"    Text: {m['text']}")

print(f"\n=== FALSE NEGATIVES ({len(fns)}) [Gold: hateful -> Pred: unhateful] ===")
for m in fns:
    print(f"[{m['id']}] Prob: {m['prob_hateful']:.4f} | Conf: {m['confidence']:.4f}")
    print(f"    Text: {m['text']}")

