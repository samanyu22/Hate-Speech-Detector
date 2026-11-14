import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer_classifier import MultilingualTransformerEngine

clf = MultilingualTransformerEngine()
clf.load_checkpoint()

sentences = [
    "i hate jews",
    "I hate Muslims",
    "I hate Hindus",
    "I hate Dalits",
    "I hate women",
    "I hate black people",
    "kill all jews",
    "jews are parasites"
]

print("=== HATE SENTENCE DIAGNOSTIC ===")
for s in sentences:
    res = clf.predict(s)
    has_ground = clf.has_grounded_markers(s)
    print(f"'{s}': label={res['label_name']} (conf={res['confidence']:.4f}, raw_prob={res['raw_hateful_probability']:.4f}, ground={has_ground}, coverage={res['content_coverage']})")
    print(f"    logits={res['logits']}")

