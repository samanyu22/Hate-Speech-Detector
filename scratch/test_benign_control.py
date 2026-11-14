import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer_classifier import MultilingualTransformerEngine

clf = MultilingualTransformerEngine()
clf.load_checkpoint()

benign_tests = [
    "I hate pollution",
    "I hate Mondays",
    "I hate winter",
    "I disagree with that government policy",
    "People from many religions live peacefully in our city.",
    "The public library has introduced new books about history."
]

print("=== BENIGN NON-HATEFUL CONTROL TESTS ===")
for t in benign_tests:
    res = clf.predict(t)
    print(f"'{t}' -> {res['label_name']} (conf={res['confidence']:.4f})")

