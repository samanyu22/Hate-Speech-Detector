import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import io
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.transformer_classifier import MultilingualTransformerEngine

# Load the user dataset from test_user_dataset.py
from scratch.test_user_dataset import raw_tsv

lines = raw_tsv.strip().split("\n")
reader = csv.DictReader(lines, delimiter="\t")
samples = []
for row in reader:
    text = row.get("Text", "").strip()
    lbl_str = row.get("Label", "").strip().lower()
    lbl = 1 if lbl_str in ["hateful", "hate", "1"] else 0
    samples.append({
        "id": row.get("ID", "").strip(),
        "text": text,
        "true_label": lbl,
        "label_str": lbl_str
    })

print(f"Total user test samples loaded from Batch 1: {len(samples)}")

engine = MultilingualTransformerEngine()
if engine.checkpoint_path.exists():
    engine.load_checkpoint()

y_true = []
y_pred = []
mismatches = []

for s in samples:
    pred = engine.predict(s["text"])
    pred_lbl = pred["label"]
    y_true.append(s["true_label"])
    y_pred.append(pred_lbl)
    
    if pred_lbl != s["true_label"]:
        mismatches.append({
            "id": s["id"],
            "text": s["text"],
            "true": "hateful" if s["true_label"] == 1 else "unhateful",
            "pred": pred["label_name"],
            "conf": pred["confidence"],
            "raw_prob": pred.get("raw_hateful_probability")
        })

acc = accuracy_score(y_true, y_pred)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_true, y_pred).tolist()

print("\n" + "=" * 60)
print(f"EVALUATION RESULTS ON USER TEST SET ({len(samples)} samples: id1 - id{len(samples)})")
print("=" * 60)
print(f"Accuracy:        {acc * 100:.2f}% ({sum(1 for t, p in zip(y_true, y_pred) if t == p)} / {len(samples)})")
print(f"Macro Precision: {prec_m * 100:.2f}%")
print(f"Macro Recall:    {rec_m * 100:.2f}%")
print(f"Macro F1-Score:  {f1_m * 100:.2f}%")
print(f"Weighted F1:     {f1_w * 100:.2f}%")
print(f"Confusion Matrix (TN, FP / FN, TP): {cm}")
print(f"Total Mismatches: {len(mismatches)}")

if mismatches:
    print("\nMismatches detail:")
    for m in mismatches[:10]:
        print(f"  [{m['id']}] True: {m['true']} | Pred: {m['pred']} ({m['conf']*100:.1f}%) | Text: {m['text']}")

# EDGE CASES EVALUATION
print("\n" + "=" * 60)
print("EDGE CASES EVALUATION")
print("=" * 60)

edge_cases = [
    # 1. Interpersonal Insults without demographic target
    {"category": "Interpersonal Insult (No demographic target)", "text": "you are absolutely stupid and have no idea what you're talking about.", "expected": 0},
    {"category": "Interpersonal Insult (No demographic target)", "text": "You are a complete idiot and a clown.", "expected": 0},
    {"category": "Interpersonal Insult (No demographic target)", "text": "Shut up, your opinion is completely worthless.", "expected": 0},
    
    # 2. Benevolent Demographic Statements (Praise/Neutral mentioning of protected groups)
    {"category": "Benevolent Identity Mention", "text": "People from Muslim community are hardworking and honest.", "expected": 0},
    {"category": "Benevolent Identity Mention", "text": "Dalits have contributed immensely to the Indian constitution and society.", "expected": 0},
    {"category": "Benevolent Identity Mention", "text": "The Sikh community runs langar to feed anyone in need regardless of religion.", "expected": 0},
    {"category": "Benevolent Identity Mention", "text": "Christians in our town are hosting a charity event for the homeless.", "expected": 0},
    
    # 3. Counter-speech / Anti-hate quotations
    {"category": "Counter-Speech / Discussion", "text": "Someone wrote 'these immigrants are vermin' which is completely acceptable racist hate speech.", "expected": 1}, # Note: user screenshot showed 'acceptable racist hate speech'
    {"category": "Counter-Speech / Discussion", "text": "We must stand united against hatred and racism toward immigrants.", "expected": 0},
    {"category": "Counter-Speech / Discussion", "text": "I strongly condemn the anti-Semitic attacks that took place yesterday.", "expected": 0},
    
    # 4. Severe Hate Speech & Slurs (True Positives)
    {"category": "Direct Hate Speech / Slur", "text": "these immigrants are vermin", "expected": 1},
    {"category": "Direct Hate Speech / Slur", "text": "kill all muslims", "expected": 1},
    {"category": "Direct Hate Speech / Slur", "text": "tu harami dalit hai yahan se bhag ja", "expected": 1},
    {"category": "Direct Hate Speech / Slur", "text": "All jews are parasites that ruin every country.", "expected": 1},
    
    # 5. Multilingual Hindi / Code-mixed Hinglish (Non-hate & Hate)
    {"category": "Hindi / Hinglish Benign", "text": "Bhai kal ka cricket match dekh ke maza aa gaya!", "expected": 0},
    {"category": "Hindi / Hinglish Benign", "text": "नमस्ते, आज का दिन बहुत सुंदर और शांत है।", "expected": 0},
    {"category": "Hindi / Hinglish Hate", "text": "ye suar mulle desh ke gaddar hain inko khatam karo", "expected": 1},
    
    # 6. Literary, Philosophical & Banter
    {"category": "Literary / Philosophical", "text": "People fake a lot of human interactions but that's just life.", "expected": 0},
    {"category": "Literary / Philosophical", "text": "I'm a very neat monster.", "expected": 0},
    {"category": "Literary / Philosophical", "text": "French cuisine is admired worldwide.", "expected": 0},
]

edge_results = []
for ec in edge_cases:
    res = engine.predict(ec["text"])
    match = (res["label"] == ec["expected"])
    edge_results.append({
        "category": ec["category"],
        "text": ec["text"],
        "expected": "HATEFUL" if ec["expected"] == 1 else "NON-HATEFUL",
        "pred": res["label_name"],
        "confidence": res["confidence"],
        "match": match
    })
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {ec['category']:<35} | Pred: {res['label_name']:<12} ({res['confidence']*100:.1f}%) | Exp: {ec['expected']} | Text: '{ec['text']}'")

edge_pass_count = sum(1 for e in edge_results if e["match"])
print(f"\nEdge Cases Passed: {edge_pass_count} / {len(edge_cases)} ({edge_pass_count/len(edge_cases)*100:.1f}%)")
