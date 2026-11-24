import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.transformer_classifier import MultilingualTransformerEngine

engine = MultilingualTransformerEngine()
if engine.checkpoint_path.exists():
    engine.load_checkpoint()

# Test Batch 2 (id211 - id530: 320 samples)
from scratch.test_batch2 import raw_data as raw2
lines2 = [l.strip() for l in raw2.strip().split("\n") if "|" in l]
samples2 = []
for l in lines2:
    parts = [p.strip() for p in l.split("|")]
    if len(parts) >= 3:
        tid, text, lbl_str = parts[0], parts[1], parts[2].lower()
        lbl = 1 if lbl_str == "hateful" else 0
        samples2.append((tid, text, lbl))

y_true2 = [s[2] for s in samples2]
y_pred2 = [engine.predict(s[1])["label"] for s in samples2]
acc2 = accuracy_score(y_true2, y_pred2)
prec2, rec2, f1_2, _ = precision_recall_fscore_support(y_true2, y_pred2, average="macro", zero_division=0)
print(f"Batch 2 ({len(samples2)} samples, id211-id530): Accuracy = {acc2*100:.2f}%, Macro F1 = {f1_2*100:.2f}%")

# Test Batch 3 (id531 - id750: 220 samples)
from scratch.test_batch3 import raw_data as raw3
lines3 = [l.strip() for l in raw3.strip().split("\n") if "|" in l]
samples3 = []
for l in lines3:
    parts = [p.strip() for p in l.split("|")]
    if len(parts) >= 3:
        tid, text, lbl_str = parts[0], parts[1], parts[2].lower()
        lbl = 1 if lbl_str == "hateful" else 0
        samples3.append((tid, text, lbl))

y_true3 = [s[2] for s in samples3]
y_pred3 = [engine.predict(s[1])["label"] for s in samples3]
acc3 = accuracy_score(y_true3, y_pred3)
prec3, rec3, f1_3, _ = precision_recall_fscore_support(y_true3, y_pred3, average="macro", zero_division=0)
print(f"Batch 3 ({len(samples3)} samples, id531-id750): Accuracy = {acc3*100:.2f}%, Macro F1 = {f1_3*100:.2f}%")

# Test Batch 4 (nid1 - nid400: 400 samples)
from scratch.test_batch4 import raw_data as raw4
lines4 = [l.strip() for l in raw4.strip().split("\n") if "|" in l]
samples4 = []
for l in lines4:
    parts = [p.strip() for p in l.split("|")]
    if len(parts) >= 3:
        tid, text, lbl_str = parts[0], parts[1], parts[2].lower()
        lbl = 1 if lbl_str == "hateful" else 0
        samples4.append((tid, text, lbl))

y_true4 = [s[2] for s in samples4]
y_pred4 = [engine.predict(s[1])["label"] for s in samples4]
acc4 = accuracy_score(y_true4, y_pred4)
prec4, rec4, f1_4, _ = precision_recall_fscore_support(y_true4, y_pred4, average="macro", zero_division=0)
print(f"Batch 4 ({len(samples4)} samples, nid1-nid400): Accuracy = {acc4*100:.2f}%, Macro F1 = {f1_4*100:.2f}%")

# Overall across all 1,150 samples
all_true = [s[2] for s in samples2] + [s[2] for s in samples3] + [s[2] for s in samples4]
all_pred = y_pred2 + y_pred3 + y_pred4
all_acc = accuracy_score(all_true, all_pred)
all_f1 = precision_recall_fscore_support(all_true, all_pred, average="macro", zero_division=0)[2]
print(f"\nCOMBINED (Batches 2-4: {len(all_true)} samples): Accuracy = {all_acc*100:.2f}%, Macro F1 = {all_f1*100:.2f}%")
