import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer_classifier import MultilingualTransformerEngine

clf = MultilingualTransformerEngine()
clf.load_checkpoint()

test_cases = [
    ("i hate jews", "HATEFUL"),
    ("i dont hate jews", "NON-HATEFUL"),
    ("i do not hate jews", "NON-HATEFUL"),
    ("we never hate muslims", "NON-HATEFUL"),
    ("I don't hate women", "NON-HATEFUL"),
    ("I hate Muslims", "HATEFUL"),
    ("I hate Hindus", "HATEFUL"),
    ("I hate Dalits", "HATEFUL"),
    ("I hate black people", "HATEFUL"),
    ("I don't hate black people", "NON-HATEFUL"),
    ("kill all jews", "HATEFUL"),
    ("we should not kill jews", "NON-HATEFUL"),
    ("nigger", "HATEFUL"),
    ("I hate pollution", "NON-HATEFUL"),
    ("I love everyone", "NON-HATEFUL")
]

print("=== NEGATION RESOLUTION PREDICTION TEST ===")
all_pass = True
for text, expected in test_cases:
    res = clf.predict(text)
    pred_label = res["label_name"]
    is_correct = (pred_label == expected)
    if not is_correct:
        all_pass = False
    print(f"'{text}' -> {pred_label} (expected: {expected}, conf={res['confidence']:.4f}, prob_h={res['raw_hateful_probability']:.4f}) [{'PASS' if is_correct else 'FAIL'}]")

print(f"\nOVERALL RESULT: {'ALL PASS!' if all_pass else 'SOME FAILED'}")

