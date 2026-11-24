from src.models.transformer_classifier import MultilingualTransformerEngine
from src.classification.classical_baseline import ClassicalBaselineClassifier

t = "you are absolutely stupid and have no idea what you're talking about."

trans = MultilingualTransformerEngine()
trans.load_checkpoint()
print("TRANSFORMER PREDICTION:")
print(trans.predict(t))

classical = ClassicalBaselineClassifier()
classical.load()
print("\nCLASSICAL TF-IDF PREDICTION:")
print(classical.predict(t))
