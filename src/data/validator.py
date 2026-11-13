from typing import List, Dict, Any
from collections import Counter
from src.data.adapter import DatasetRecord

class DatasetValidator:
    @staticmethod
    def validate(records: List[DatasetRecord]) -> Dict[str, Any]:
        if not records:
            return {
                "is_valid": False,
                "sample_count": 0,
                "class_distribution": {},
                "missing_text_count": 0,
                "missing_image_count": 0,
                "duplicate_count": 0,
                "detected_languages": {},
                "warnings": ["Dataset contains zero valid records."],
                "report_summary": "Validation failed: Dataset is empty."
            }

        total_samples = len(records)
        class_dist = Counter()
        missing_text = 0
        missing_images = 0
        texts_seen = set()
        duplicate_count = 0
        languages_count = Counter()
        warnings = []

        for rec in records:
            class_dist[str(rec.label)] += 1
            if not rec.text or not rec.text.strip():
                missing_text += 1

            norm_text = rec.text.strip().lower()
            if norm_text in texts_seen:
                duplicate_count += 1
            else:
                texts_seen.add(norm_text)

            if rec.language:
                languages_count[rec.language] += 1
            else:
                languages_count["unspecified"] += 1

        # Check balance
        hateful_count = class_dist.get("1", 0)
        non_hateful_count = class_dist.get("0", 0)
        if total_samples > 0:
            hate_ratio = hateful_count / total_samples
            if hate_ratio < 0.10 or hate_ratio > 0.90:
                warnings.append(f"Severe class imbalance: Hateful class is {hate_ratio*100:.1f}% of total samples.")

        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate text samples. Check split boundaries.")

        if missing_text > 0:
            warnings.append(f"Found {missing_text} samples with empty text.")

        is_valid = (total_samples > 0 and (hateful_count + non_hateful_count == total_samples) and missing_text < total_samples)

        summary = (
            f"Dataset Validation: {total_samples} samples inspected. "
            f"Class 0 (Non-Hateful): {non_hateful_count}, Class 1 (Hateful): {hateful_count}. "
            f"Duplicates: {duplicate_count}. Warnings: {len(warnings)}."
        )

        return {
            "is_valid": is_valid,
            "sample_count": total_samples,
            "class_distribution": dict(class_dist),
            "missing_text_count": missing_text,
            "missing_image_count": missing_images,
            "duplicate_count": duplicate_count,
            "detected_languages": dict(languages_count),
            "warnings": warnings,
            "report_summary": summary
        }

