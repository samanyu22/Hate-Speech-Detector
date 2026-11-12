"""
Dataset Adapter Module
Provides standardized dataset loading, schema normalization, and validation
for external datasets (FaceForensics++, DFDC, FakeAVCeleb, Fakeddit) and local manifests.
"""

import os
import json
import csv
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class DatasetRecord:
    """Represents a single standardized multimodal data sample."""
    def __init__(
        self,
        sample_id: str,
        label: str,  # "REAL" or "FAKE" (or "UNCERTAIN")
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        manipulation_type: str = "unknown",
        source: str = "unknown",
        split: str = "train",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.sample_id = str(sample_id)
        self.label = label.upper().strip()
        self.text = text or ""
        self.image_path = image_path
        self.audio_path = audio_path
        self.video_path = video_path
        self.manipulation_type = manipulation_type
        self.source = source
        self.split = split
        self.metadata = metadata or {}

    @property
    def available_modalities(self) -> List[str]:
        mods = []
        if self.text and len(self.text.strip()) > 0:
            mods.append("text")
        if self.image_path and os.path.exists(self.image_path):
            mods.append("image")
        if self.audio_path and os.path.exists(self.audio_path):
            mods.append("audio")
        if self.video_path and os.path.exists(self.video_path):
            mods.append("video")
        return mods

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "text": self.text,
            "image_path": self.image_path,
            "audio_path": self.audio_path,
            "video_path": self.video_path,
            "manipulation_type": self.manipulation_type,
            "source": self.source,
            "split": self.split,
            "available_modalities": self.available_modalities,
            "metadata": self.metadata
        }


class DatasetAdapter:
    """Adapts arbitrary JSON or CSV manifests into standardized DatasetRecord lists."""

    @staticmethod
    def load_from_json(file_path: str, base_dir: Optional[str] = None) -> List[DatasetRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {file_path}")

        base = Path(base_dir) if base_dir else path.parent

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records_raw = data if isinstance(data, list) else data.get("samples", [])
        records = []
        for item in records_raw:
            img = item.get("image_path")
            if img and not os.path.isabs(img):
                img = str((base / img).resolve())

            aud = item.get("audio_path")
            if aud and not os.path.isabs(aud):
                aud = str((base / aud).resolve())

            vid = item.get("video_path")
            if vid and not os.path.isabs(vid):
                vid = str((base / vid).resolve())

            rec = DatasetRecord(
                sample_id=item.get("id", item.get("sample_id", f"sample_{len(records)}")),
                label=item.get("label", "UNCERTAIN"),
                text=item.get("text", item.get("caption", item.get("transcript", ""))),
                image_path=img,
                audio_path=aud,
                video_path=vid,
                manipulation_type=item.get("manipulation_type", "unknown"),
                source=item.get("source", "manifest"),
                split=item.get("split", "train"),
                metadata=item.get("metadata", {})
            )
            records.append(rec)
        return records

    @staticmethod
    def load_from_csv(file_path: str, base_dir: Optional[str] = None) -> List[DatasetRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {file_path}")

        base = Path(base_dir) if base_dir else path.parent
        records = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                img = row.get("image_path") or row.get("image") or None
                if img and not os.path.isabs(img):
                    img = str((base / img).resolve())

                aud = row.get("audio_path") or row.get("audio") or None
                if aud and not os.path.isabs(aud):
                    aud = str((base / aud).resolve())

                vid = row.get("video_path") or row.get("video") or None
                if vid and not os.path.isabs(vid):
                    vid = str((base / vid).resolve())

                rec = DatasetRecord(
                    sample_id=row.get("sample_id") or row.get("id") or f"csv_sample_{i}",
                    label=row.get("label", "UNCERTAIN"),
                    text=row.get("text") or row.get("caption") or row.get("transcript") or "",
                    image_path=img,
                    audio_path=aud,
                    video_path=vid,
                    manipulation_type=row.get("manipulation_type", "unknown"),
                    source=row.get("source", "csv_import"),
                    split=row.get("split", "train")
                )
                records.append(rec)
        return records

