import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.data.adapter import UniversalDatasetAdapter
from src.models.transformer_classifier import MultilingualTransformerEngine
from scratch.test_user_dataset import data

print("Loading datasets...")
adapter_hasoc = UniversalDatasetAdapter(
    file_path="data/raw/hasoc_hi_en.jsonl",
    text_column="text",
    label_column="label",
    language_column="language"
)
records_hasoc = adapter_hasoc.load()

adapter_hx = UniversalDatasetAdapter(
    file_path="data/raw/hatexplain_benchmark.jsonl",
    text_column="post_tokens",
    label_column="label"
)
records_hx = adapter_hx.load()

adapter_subtle = UniversalDatasetAdapter(
    file_path="data/raw/subtle_structural_benchmark.jsonl",
    text_column="text",
    label_column="label",
    language_column="language"
)
records_subtle = adapter_subtle.load()

all_train_records = records_hasoc + records_hx + records_subtle
print(f"Total Combined Training Records: {len(all_train_records)} (HASOC: {len(records_hasoc)}, HateXplain: {len(records_hx)}, Subtle: {len(records_subtle)})")

print("Training Multilingual Transformer with subtle & structural speech augmentations...")
engine = MultilingualTransformerEngine()
train_stats = engine.train(all_train_records, epochs=15, lr=2e-3)
print(f"Training Complete! Final Loss: {train_stats['final_loss']:.4f}, Vocab Size: {train_stats['vocab_size']}")

print("\nSaving updated checkpoint...")
engine.save_checkpoint()

print("\nEvaluating on 210-sample user test dataset...")
y_true = []
y_pred = []
results = []

for row in data:
    sample_id = row["ID"]
    text = row["Text"]
    gold_str = row["Label"].strip().lower()
    gold_label = 1 if gold_str == "hateful" else 0
    
    pred_res = engine.predict(text)
    pred_label = pred_res["label"]
    pred_name = "hateful" if pred_label == 1 else "unhateful"
    confidence = pred_res["confidence"]
    raw_h_prob = pred_res["raw_hateful_probability"]
    
    y_true.append(gold_label)
    y_pred.append(pred_label)
    
    is_match = (gold_label == pred_label)
    results.append({
        "id": sample_id,
        "text": text,
        "gold_label": gold_str,
        "gold_numeric": gold_label,
        "pred_label": pred_name,
        "pred_numeric": pred_label,
        "confidence": confidence,
        "prob_hateful": raw_h_prob,
        "match": is_match
    })

acc = accuracy_score(y_true, y_pred)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

eval_summary = {
    "total_samples": len(data),
    "accuracy": round(float(acc), 4),
    "macro_f1": round(float(f1_m), 4),
    "macro_precision": round(float(prec_m), 4),
    "macro_recall": round(float(rec_m), 4),
    "confusion_matrix": {
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp)
    },
    "per_class": {
        "unhateful": {
            "precision": round(float(prec_cls[0]), 4),
            "recall": round(float(rec_cls[0]), 4),
            "f1": round(float(f1_cls[0]), 4),
            "support": int(support[0])
        },
        "hateful": {
            "precision": round(float(prec_cls[1]), 4),
            "recall": round(float(rec_cls[1]), 4),
            "f1": round(float(f1_cls[1]), 4),
            "support": int(support[1])
        }
    },
    "mismatches": [r for r in results if not r["match"]],
    "all_results": results
}

with open("scratch/eval_results_after_retrain.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)

print("\n" + "="*50)
print("RETRAINED MODEL PERFORMANCE ON BENCHMARK:")
print(f"Accuracy: {acc*100:.2f}% (was 52.38%)")
print(f"Macro F1: {f1_m:.4f} (was 0.3842)")
print(f"Macro Precision: {prec_m:.4f} | Macro Recall: {rec_m:.4f}")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Remaining Mismatches: {len(eval_summary['mismatches'])} (down from 100)")
print("="*50)

