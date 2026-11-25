import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.transformer_classifier import MultilingualTransformerEngine

def run_1150_benchmark(checkpoint_path=None):
    engine = MultilingualTransformerEngine()
    if checkpoint_path:
        engine.load_checkpoint(checkpoint_path)
    elif engine.checkpoint_path.exists():
        engine.load_checkpoint()

    # Batch 1 (from test_user_dataset)
    from scratch.test_user_dataset import raw_tsv as raw1
    lines1 = [l.strip() for l in raw1.strip().split("\n")[1:] if "\t" in l]
    samples1 = []
    for l in lines1:
        parts = [p.strip() for p in l.split("\t")]
        if len(parts) >= 3:
            tid, text, lbl_str = parts[0], parts[1], parts[2].lower()
            lbl = 1 if lbl_str == "hateful" else 0
            samples1.append((tid, text, lbl))

    # Batch 2
    from scratch.test_batch2 import raw_data as raw2
    lines2 = [l.strip() for l in raw2.strip().split("\n") if "|" in l]
    samples2 = []
    for l in lines2:
        parts = [p.strip() for p in l.split("|")]
        if len(parts) >= 3:
            tid, text, lbl_str = parts[0], parts[1], parts[2].lower()
            lbl = 1 if lbl_str == "hateful" else 0
            samples2.append((tid, text, lbl))

    # Batch 3
    from scratch.test_batch3 import raw_data as raw3
    lines3 = [l.strip() for l in raw3.strip().split("\n") if "|" in l]
    samples3 = []
    for l in lines3:
        parts = [p.strip() for p in l.split("|")]
        if len(parts) >= 3:
            tid, text, lbl_str = parts[0], parts[1], parts[2].lower()
            lbl = 1 if lbl_str == "hateful" else 0
            samples3.append((tid, text, lbl))

    # Batch 4
    from scratch.test_batch4 import raw_data as raw4
    lines4 = [l.strip() for l in raw4.strip().split("\n") if "|" in l]
    samples4 = []
    for l in lines4:
        parts = [p.strip() for p in l.split("|")]
        if len(parts) >= 3:
            tid, text, lbl_str = parts[0], parts[1], parts[2].lower()
            lbl = 1 if lbl_str == "hateful" else 0
            samples4.append((tid, text, lbl))

    def eval_set(samples, name):
        y_t = [s[2] for s in samples]
        y_p = [engine.predict(s[1])["label"] for s in samples]
        acc = accuracy_score(y_t, y_p)
        prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_p, average="macro", zero_division=0)
        cm = confusion_matrix(y_t, y_p)
        print(f"{name} ({len(samples)} samples): Accuracy = {acc*100:.2f}%, Macro F1 = {f1*100:.2f}% (TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]})")
        return y_t, y_p

    print("=" * 60)
    print("RUNNING 1,150-SAMPLE BENCHMARK EVALUATION")
    print("=" * 60)
    t1, p1 = eval_set(samples1, "Batch 1 (Subtle & Structural)")
    t2, p2 = eval_set(samples2, "Batch 2 (Multilingual HASOC/Slang)")
    t3, p3 = eval_set(samples3, "Batch 3 (Conversational/Context)")
    t4, p4 = eval_set(samples4, "Batch 4 (Large Real-World)")

    all_t = t1 + t2 + t3 + t4
    all_p = p1 + p2 + p3 + p4
    all_acc = accuracy_score(all_t, all_p)
    all_prec, all_rec, all_f1, _ = precision_recall_fscore_support(all_t, all_p, average="macro", zero_division=0)
    all_cm = confusion_matrix(all_t, all_p)

    print("-" * 60)
    print(f"COMBINED 1,150 SAMPLES BENCHMARK SUMMARY:")
    print(f"Accuracy:  {all_acc*100:.2f}%")
    print(f"Macro F1:  {all_f1*100:.2f}%")
    print(f"Precision: {all_prec*100:.2f}%")
    print(f"Recall:    {all_rec*100:.2f}%")
    print(f"Confusion Matrix: TN={all_cm[0][0]}, FP={all_cm[0][1]}, FN={all_cm[1][0]}, TP={all_cm[1][1]}")
    print("=" * 60)

    return {
        "accuracy": all_acc,
        "macro_f1": all_f1,
        "precision": all_prec,
        "recall": all_rec,
        "confusion_matrix": all_cm.tolist()
    }

if __name__ == "__main__":
    run_1150_benchmark()
