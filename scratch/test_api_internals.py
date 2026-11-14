import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.api.detection import get_models, detect_content
from backend.schemas.detection import DetectionRequest

classical_clf, transformer_clf = get_models()
print("transformer_clf:", transformer_clf)
print("is_trained:", transformer_clf.is_trained)
pred = transformer_clf.predict("nigger")
print("Direct predict:", pred)

req = DetectionRequest(text="nigger")
api_res = detect_content(req)
print("detect_content:", api_res.predicted_label, api_res.label_name, api_res.confidence)
