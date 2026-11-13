import json
from pathlib import Path
from backend.core.config import settings
from backend.database.db import get_connection
from src.data.adapter import UniversalDatasetAdapter
from src.data.validator import DatasetValidator

def register_default_datasets():
    conn = get_connection()
    cursor = conn.cursor()
    base_dir = settings.base_dir

    configs = [
        {
            "id": "hasoc_hi_en",
            "name": "HASOC Hindi-English Code-Mixed Corpus",
            "source": "FIRE Track on Hate Speech and Offensive Content",
            "license": "Research Use Only",
            "languages": ["hi", "en", "hi-en"],
            "modalities": ["text"],
            "file_path": str(base_dir / "data" / "raw" / "hasoc_hi_en.jsonl"),
            "text_column": "text",
            "label_column": "label",
            "label_mapping": {"0": 0, "1": 1}
        },
        {
            "id": "hatexplain_benchmark",
            "name": "HateXplain Benchmark Subset",
            "source": "Mathew et al. (AAAI 2021)",
            "license": "MIT License",
            "languages": ["en"],
            "modalities": ["text", "rationales"],
            "file_path": str(base_dir / "data" / "raw" / "hatexplain_benchmark.jsonl"),
            "text_column": "post_tokens",
            "label_column": "label",
            "label_mapping": {"0": 0, "1": 1}
        },
        {
            "id": "multimodal_memes",
            "name": "Multimodal Code-Mixed Social Meme Corpus",
            "source": "Multimodal Hate Research Project",
            "license": "Research Non-Commercial",
            "languages": ["en", "hi", "hi-en"],
            "modalities": ["text", "image", "ocr", "context"],
            "file_path": str(base_dir / "data" / "raw" / "multimodal_memes.jsonl"),
            "text_column": "caption",
            "label_column": "label",
            "image_column": "image_path",
            "context_column": "context",
            "label_mapping": {"0": 0, "1": 1}
        }
    ]

    for cfg in configs:
        adapter = UniversalDatasetAdapter(
            file_path=cfg["file_path"],
            text_column=cfg["text_column"],
            label_column=cfg["label_column"],
            image_column=cfg.get("image_column"),
            context_column=cfg.get("context_column"),
            label_mapping=cfg["label_mapping"]
        )
        records = adapter.load()
        val_report = DatasetValidator.validate(records)

        cursor.execute("""
            INSERT OR REPLACE INTO datasets 
            (id, name, source, license, languages, modalities, file_path, sample_count, class_distribution, validation_report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cfg["id"],
            cfg["name"],
            cfg["source"],
            cfg["license"],
            json.dumps(cfg["languages"]),
            json.dumps(cfg["modalities"]),
            cfg["file_path"],
            val_report["sample_count"],
            json.dumps(val_report["class_distribution"]),
            json.dumps(val_report)
        ))
        print(f"Registered {cfg['id']}: {val_report['sample_count']} samples, Valid: {val_report['is_valid']}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    register_default_datasets()

