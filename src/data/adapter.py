import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

class DatasetRecord:
    def __init__(
        self,
        id: str,
        text: str,
        label: int,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(id)
        self.text = str(text) if text is not None else ""
        self.label = int(label)
        self.image_path = str(image_path) if image_path else None
        self.context = str(context) if context else None
        self.language = str(language) if language else None
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "label_name": "HATEFUL" if self.label == 1 else "NON-HATEFUL",
            "image_path": self.image_path,
            "context": self.context,
            "language": self.language or "en",
            "metadata": self.metadata
        }
        if "ocr_text" in self.metadata:
            d["ocr_text"] = self.metadata["ocr_text"]
        if "category" in self.metadata:
            d["category"] = self.metadata["category"]
        if "target" in self.metadata:
            d["target"] = self.metadata["target"]
        return d

class UniversalDatasetAdapter:
    def __init__(
        self,
        file_path: str,
        text_column: str = "text",
        label_column: str = "label",
        image_column: Optional[str] = None,
        context_column: Optional[str] = None,
        language_column: Optional[str] = None,
        id_column: Optional[str] = "id",
        label_mapping: Optional[Dict[str, int]] = None
    ):
        self.file_path = Path(file_path)
        self.text_column = text_column
        self.label_column = label_column
        self.image_column = image_column
        self.context_column = context_column
        self.language_column = language_column
        self.id_column = id_column
        self.label_mapping = label_mapping or {"0": 0, "1": 1, 0: 0, 1: 1}

    def load(self) -> List[DatasetRecord]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        records: List[DatasetRecord] = []
        suffix = self.file_path.suffix.lower()

        if suffix == ".jsonl":
            with open(self.file_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    rec = self._parse_item(item, default_id=f"rec_{idx}")
                    if rec:
                        records.append(rec)

        elif suffix == ".json":
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.get("records", [])
                for idx, item in enumerate(items):
                    rec = self._parse_item(item, default_id=f"rec_{idx}")
                    if rec:
                        records.append(rec)

        elif suffix in [".csv", ".tsv"]:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(self.file_path, sep=sep, encoding="utf-8")
            for idx, row in df.iterrows():
                rec = self._parse_item(row.to_dict(), default_id=f"rec_{idx}")
                if rec:
                    records.append(rec)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

        return records

    def _parse_item(self, item: Dict[str, Any], default_id: str) -> Optional[DatasetRecord]:
        text = item.get(self.text_column)
        if text is None or text == "":
            for alt in ["post_tokens", "caption", "sentence", "content", "tweet", "ocr_text"]:
                if alt in item and item[alt]:
                    text = item[alt]
                    break
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        if text is None:
            text = ""

        raw_label = item.get(self.label_column)
        if raw_label is None:
            for alt in ["label", "gold_label", "target_label", "annotation"]:
                if alt in item and item[alt] is not None:
                    raw_label = item[alt]
                    break
        if raw_label is None:
            return None

        # Map label
        str_key = str(raw_label).strip()
        int_key = None
        try:
            int_key = int(raw_label)
        except (ValueError, TypeError):
            pass

        label = None
        if raw_label in self.label_mapping:
            label = self.label_mapping[raw_label]
        elif str_key in self.label_mapping:
            label = self.label_mapping[str_key]
        elif int_key is not None and int_key in self.label_mapping:
            label = self.label_mapping[int_key]
        else:
            # Try boolean or direct 0/1
            if str_key.lower() in ["true", "hate", "hateful", "hof"]:
                label = 1
            elif str_key.lower() in ["false", "non-hate", "non_hate", "not", "normal"]:
                label = 0
            else:
                return None

        rec_id = item.get(self.id_column, default_id)
        image_path = item.get(self.image_column) if self.image_column else None
        if not image_path:
            for alt in ["image_path", "image", "img_path", "img"]:
                if alt in item and item[alt]:
                    image_path = item[alt]
                    break

        context = item.get(self.context_column) if self.context_column else None
        if not context:
            for alt in ["context", "thread", "history", "thread_history"]:
                if alt in item and item[alt]:
                    context = item[alt]
                    break

        language = item.get(self.language_column) if self.language_column else None
        if not language:
            for alt in ["language", "lang", "lang_code"]:
                if alt in item and item[alt]:
                    language = item[alt]
                    break

        # Remaining fields go into metadata
        excluded = {self.text_column, self.label_column, self.image_column, self.context_column, self.language_column, self.id_column}
        metadata = {k: v for k, v in item.items() if k not in excluded}

        return DatasetRecord(
            id=str(rec_id),
            text=str(text),
            label=label,
            image_path=image_path,
            context=context,
            language=language,
            metadata=metadata
        )

