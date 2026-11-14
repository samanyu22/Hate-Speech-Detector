import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer_classifier import MultilingualTransformerEngine

clf = MultilingualTransformerEngine()
clf.load_checkpoint()

print("1. 'i hate jews':", clf.predict("i hate jews"))
print("2. 'hate jews':", clf.predict("hate jews"))
print("3. 'all jews should die':", clf.predict("all jews should die"))

