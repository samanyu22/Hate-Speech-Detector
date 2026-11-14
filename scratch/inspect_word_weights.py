import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.transformer_classifier import MultilingualTransformerEngine

clf = MultilingualTransformerEngine()
clf.load_checkpoint()

words = ["i", "hate", "jews", "muslims", "hindus", "women", "food", "movie"]
for w in words:
    idx = clf.vocab.get(w, clf.vocab["<UNK>"])
    is_known = w in clf.vocab
    x = clf.tokenize_to_tensor(w)
    with torch.no_grad():
        logits = clf.model(x)[0].cpu().numpy()
    print(f"Word: '{w}' -> in_vocab: {is_known}, idx: {idx}, logits: [z0={logits[0]:.4f}, z1={logits[1]:.4f}]")

