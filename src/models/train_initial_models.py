import json
from pathlib import Path
from backend.core.config import settings
from backend.database.db import get_connection
from src.data.adapter import UniversalDatasetAdapter
from src.data.splitter import DatasetSplitter
from src.classification.classical_baseline import ClassicalBaselineClassifier
from src.models.transformer_classifier import MultilingualTransformerEngine

def train_and_register_baselines():
    conn = get_connection()
    cursor = conn.cursor()

    # Load Multilingual Datasets (HASOC + HateXplain)
    hasoc_path = settings.base_dir / "data" / "raw" / "hasoc_hi_en.jsonl"
    adapter_hasoc = UniversalDatasetAdapter(
        file_path=str(hasoc_path),
        text_column="text",
        label_column="label",
        language_column="language"
    )
    records_hasoc = adapter_hasoc.load()

    hatexplain_path = settings.base_dir / "data" / "raw" / "hatexplain_benchmark.jsonl"
    records_hx = []
    if hatexplain_path.exists():
        adapter_hx = UniversalDatasetAdapter(
            file_path=str(hatexplain_path),
            text_column="post_tokens",
            label_column="label"
        )
        records_hx = adapter_hx.load()

    all_records = records_hasoc + records_hx
    train_records, val_records, test_records = DatasetSplitter.split(all_records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    print(f"Combined Dataset split: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test")

    # 1. Train Classical Baseline
    print("Training Classical TF-IDF + Logistic Regression Baseline...")
    classical_clf = ClassicalBaselineClassifier()
    train_res = classical_clf.train(train_records)
    eval_res = classical_clf.evaluate(test_records)
    print("Classical Test Evaluation:", eval_res)

    cursor.execute("""
        INSERT OR REPLACE INTO model_registry
        (id, name, architecture, version, checkpoint_path, status, is_active, metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "classical-tfidf-svm",
        "TF-IDF + Linear Classifier Baseline",
        "classical",
        "1.0",
        str(classical_clf.checkpoint_path),
        "ready",
        0,
        json.dumps(eval_res)
    ))

    # 2. Train Multilingual Transformer
    print("Training Multilingual Code-Mixed Transformer Classifier...")
    transformer_engine = MultilingualTransformerEngine()
    t_train_res = transformer_engine.train(train_records, epochs=12, lr=2e-3)
    t_eval_res = transformer_engine.evaluate(test_records)
    print("Transformer Test Evaluation:", t_eval_res)

    cursor.execute("""
        INSERT OR REPLACE INTO model_registry
        (id, name, architecture, version, checkpoint_path, status, is_active, metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "multilingual-transformer-v1",
        "Multilingual Code-Mixed Transformer Classifier",
        "transformer",
        "1.0",
        str(transformer_engine.checkpoint_path),
        "ready",
        1, # Active model by default
        json.dumps(t_eval_res)
    ))

    # Also register version 1.0 in model_versions table
    cursor.execute("""
        INSERT OR REPLACE INTO model_versions
        (version_id, model_id, parent_version, optimization_method, reward_score, macro_f1, accuracy, checkpoint_path, eval_details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "multilingual-transformer-v1.0",
        "multilingual-transformer-v1",
        None,
        "Supervised Baseline",
        0.75,
        t_eval_res["macro_f1"],
        t_eval_res["accuracy"],
        str(transformer_engine.checkpoint_path),
        json.dumps(t_eval_res)
    ))

    conn.commit()
    conn.close()
    print("Models trained and registered successfully.")

if __name__ == "__main__":
    train_and_register_baselines()

