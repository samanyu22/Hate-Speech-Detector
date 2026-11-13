import pytest
from pathlib import Path
from src.data.adapter import UniversalDatasetAdapter
from src.data.validator import DatasetValidator
from src.data.splitter import DatasetSplitter
from backend.core.config import settings

def test_dataset_loading_and_validation():
    hasoc_path = settings.base_dir / "data" / "raw" / "hasoc_hi_en.jsonl"
    adapter = UniversalDatasetAdapter(
        file_path=str(hasoc_path),
        text_column="text",
        label_column="label"
    )
    records = adapter.load()
    assert len(records) > 0

    val = DatasetValidator.validate(records)
    assert val["is_valid"] is True
    assert val["sample_count"] == len(records)
    assert "0" in val["class_distribution"]
    assert "1" in val["class_distribution"]

def test_dataset_splitting_stratification():
    hasoc_path = settings.base_dir / "data" / "raw" / "hasoc_hi_en.jsonl"
    adapter = UniversalDatasetAdapter(str(hasoc_path))
    records = adapter.load()
    train, val, test = DatasetSplitter.split(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(train) + len(val) + len(test) == len(records)
    assert len(train) > len(val)

