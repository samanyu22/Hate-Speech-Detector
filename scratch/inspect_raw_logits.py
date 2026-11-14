import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import torch.nn.functional as F
from src.models.transformer_classifier import MultilingualTransformerEngine
from scratch.test_user_dataset import data

clf = MultilingualTransformerEngine()
clf.load_checkpoint()
clf.model.eval()

raw_preds = []
for row in data:
    sample_id = row["ID"]
    text = row["Text"]
    gold_str = row["Label"].strip().lower()
    gold_label = 1 if gold_str == "hateful" else 0
    
    x = clf.tokenize_to_tensor(text)
    with torch.no_grad():
        logits = clf.model(x)[0].cpu().numpy()
    
    z0, z1 = float(logits[0]), float(logits[1])
    raw_prob_h = float(F.softmax(torch.tensor([z0, z1]), dim=0)[1])
    raw_pred_label = 1 if raw_prob_h >= 0.5 else 0
    has_ground = clf.has_grounded_markers(text)
    
    raw_preds.append({
        "id": sample_id,
        "text": text,
        "gold": gold_str,
        "gold_label": gold_label,
        "z0": z0,
        "z1": z1,
        "raw_prob_h": raw_prob_h,
        "raw_pred_label": raw_pred_label,
        "has_ground": has_ground
    })

raw_tp = sum(1 for r in raw_preds if r["gold_label"] == 1 and r["raw_pred_label"] == 1)
raw_fp = sum(1 for r in raw_preds if r["gold_label"] == 0 and r["raw_pred_label"] == 1)
raw_tn = sum(1 for r in raw_preds if r["gold_label"] == 0 and r["raw_pred_label"] == 0)
raw_fn = sum(1 for r in raw_preds if r["gold_label"] == 1 and r["raw_pred_label"] == 0)
raw_acc = (raw_tp + raw_tn) / len(raw_preds)

print(f"RAW Neural Model (Without Grounding Guardrail):")
print(f"Accuracy: {raw_acc*100:.2f}%")
print(f"TP: {raw_tp}, FP: {raw_fp}, TN: {raw_tn}, FN: {raw_fn}")

print("\nSamples where raw neural network got it RIGHT (hateful), but grounding suppressed it:")
suppressed = [r for r in raw_preds if r["gold_label"] == 1 and r["raw_pred_label"] == 1 and not r["has_ground"]]
print(f"Count of suppressed true positives: {len(suppressed)}")
for r in suppressed[:10]:
    print(f"[{r['id']}] raw_prob={r['raw_prob_h']:.4f} | {r['text']}")
