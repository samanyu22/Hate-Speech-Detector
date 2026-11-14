import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer_classifier import MultilingualTransformerEngine

clf = MultilingualTransformerEngine()
clf.load_checkpoint()

test_texts = [
    "You are a nigger",
    "I hate niggers",
    "All niggers should die",
    "nigger",
    "sniggering at the joke",
    "The library has new books about history.",
    "chink",
    "faggot",
    "kike"
]

print("=== SLUR CLASSIFICATION TEST ===")
for t in test_texts:
    res = clf.predict(t)
    print(f"'{t}' -> {res['label_name']} (conf={res['confidence']:.4f}, prob_h={res['raw_hateful_probability']:.4f})")

