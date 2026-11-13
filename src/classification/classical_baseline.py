import os
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.preprocessing.text.normalizer import TextNormalizer
from src.data.adapter import DatasetRecord

class ClassicalBaselineClassifier:
    def __init__(self, checkpoint_path: str = "models/checkpoints/classical_baseline.joblib"):
        self.checkpoint_path = Path(checkpoint_path)
        self.normalizer = TextNormalizer()
        
        # Word + Character n-grams for robust subword and code-mixed representation
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=6000,
            sublinear_tf=True
        )
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=8000,
            sublinear_tf=True
        )
        union = FeatureUnion([
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer)
        ])
        
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.5,
            random_state=42
        )
        
        self.pipeline = Pipeline([
            ("features", union),
            ("classifier", clf)
        ])
        self.is_trained = False

    def train(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        texts = [self.normalizer.normalize(r.text) for r in records]
        labels = [r.label for r in records]

        self.pipeline.fit(texts, labels)
        self.is_trained = True

        # Save checkpoint
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.checkpoint_path)

        train_preds = self.pipeline.predict(texts)
        acc = accuracy_score(labels, train_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, train_preds, average="macro", zero_division=0)

        return {
            "train_samples": len(records),
            "train_accuracy": round(float(acc), 4),
            "train_macro_f1": round(float(f1), 4),
            "checkpoint": str(self.checkpoint_path)
        }

    def evaluate(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        if not self.is_trained:
            self.load()

        texts = [self.normalizer.normalize(r.text) for r in records]
        labels = [r.label for r in records]

        preds = self.pipeline.predict(texts)
        probs = self.pipeline.predict_proba(texts)[:, 1]

        acc = accuracy_score(labels, preds)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        cm = confusion_matrix(labels, preds).tolist()

        return {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_w), 4),
            "per_class_f1": {
                "non_hateful": round(float(f1_c[0]), 4) if len(f1_c) > 0 else 0.0,
                "hateful": round(float(f1_c[1]), 4) if len(f1_c) > 1 else 0.0
            },
            "confusion_matrix": cm,
            "sample_count": len(records)
        }

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_trained:
            self.load()

        clean_text = self.normalizer.normalize(text)
        prob_hateful = float(self.pipeline.predict_proba([clean_text])[0][1])
        pred_label = 1 if prob_hateful >= 0.50 else 0
        confidence = prob_hateful if pred_label == 1 else (1.0 - prob_hateful)

        return {
            "label": pred_label,
            "label_name": "HATEFUL" if pred_label == 1 else "NON-HATEFUL",
            "confidence": round(confidence, 4),
            "raw_hateful_probability": round(prob_hateful, 4)
        }

    def load(self):
        if self.checkpoint_path.exists():
            self.pipeline = joblib.load(self.checkpoint_path)
            self.is_trained = True
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {self.checkpoint_path}")

