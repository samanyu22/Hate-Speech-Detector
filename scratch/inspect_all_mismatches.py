import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
from scratch.test_user_dataset import raw_tsv
from src.models.transformer_classifier import MultilingualTransformerEngine

lines = raw_tsv.strip().split("\n")
reader = csv.DictReader(lines, delimiter="\t")
engine = MultilingualTransformerEngine()
if engine.checkpoint_path.exists():
    engine.load_checkpoint()

mismatches = []
for row in reader:
    text = row.get("Text", "").strip()
    lbl_str = row.get("Label", "").strip().lower()
    lbl = 1 if lbl_str in ["hateful", "hate", "1"] else 0
    pred = engine.predict(text)
    if pred["label"] != lbl:
        mismatches.append((row.get("ID"), text, lbl_str, pred["label_name"], pred["confidence"]))

print(f"Total mismatches: {len(mismatches)}")
for tid, text, true_lbl, pred_lbl, conf in mismatches:
    print(f"[{tid}] True: {true_lbl:<8} | Pred: {pred_lbl:<12} ({conf*100:.1f}%) | Text: {text}")
