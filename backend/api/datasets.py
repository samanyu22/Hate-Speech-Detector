import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from backend.schemas.dataset import DatasetRegisterRequest, DatasetValidationResponse
from backend.database.db import get_connection
from src.data.adapter import UniversalDatasetAdapter
from src.data.validator import DatasetValidator

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

@router.get("/")
def list_datasets():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        if d.get("languages"):
            d["languages"] = json.loads(d["languages"])
        if d.get("modalities"):
            d["modalities"] = json.loads(d["modalities"])
        if d.get("class_distribution"):
            d["class_distribution"] = json.loads(d["class_distribution"])
        if d.get("validation_report"):
            d["validation_report"] = json.loads(d["validation_report"])
        result.append(d)
    return result

@router.post("/register")
def register_dataset(req: DatasetRegisterRequest):
    try:
        adapter = UniversalDatasetAdapter(
            file_path=req.file_path,
            text_column=req.text_column,
            label_column=req.label_column,
            image_column=req.image_column,
            context_column=req.context_column,
            label_mapping=req.label_mapping
        )
        records = adapter.load()
        val_report = DatasetValidator.validate(records)

        conn = get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO datasets
            (id, name, source, license, languages, modalities, file_path, sample_count, class_distribution, validation_report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            req.id,
            req.name,
            req.source,
            req.license,
            json.dumps(req.languages),
            json.dumps(req.modalities),
            req.file_path,
            val_report["sample_count"],
            json.dumps(val_report["class_distribution"]),
            json.dumps(val_report)
        ))
        conn.commit()
        conn.close()

        return {"status": "success", "id": req.id, "validation": val_report}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{dataset_id}/inspect")
def inspect_dataset(
    dataset_id: str,
    limit: int = 50,
    offset: int = 0,
    label_filter: str = "all",
    search: str = ""
):
    conn = get_connection()
    row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = row["file_path"]
    try:
        adapter = UniversalDatasetAdapter(file_path=file_path)
        records = adapter.load()

        languages = json.loads(row["languages"]) if row["languages"] else []
        modalities = json.loads(row["modalities"]) if row["modalities"] else []
        class_dist = json.loads(row["class_distribution"]) if row["class_distribution"] else {}

        all_samples = [r.to_dict() for r in records]

        # Filter by label if requested
        if label_filter and label_filter.strip().lower() not in ["", "all", "none"]:
            try:
                target_label = int(label_filter.strip())
                all_samples = [s for s in all_samples if s.get("label") == target_label]
            except ValueError:
                pass

        # Filter by search term if requested
        if search and search.strip():
            query = search.strip().lower()
            all_samples = [
                s for s in all_samples
                if query in s.get("text", "").lower()
                or query in s.get("id", "").lower()
                or query in str(s.get("ocr_text", "")).lower()
                or query in str(s.get("category", "")).lower()
                or query in str(s.get("context", "")).lower()
            ]

        total_filtered = len(all_samples)
        paged_samples = all_samples[offset : offset + limit]

        # Attach image_url if image_path exists
        for s in paged_samples:
            if s.get("image_path"):
                p = Path(s["image_path"])
                s["image_url"] = f"/data/raw/images/{p.name}"

        return {
            "dataset_id": dataset_id,
            "name": row["name"],
            "source": row["source"],
            "license": row["license"],
            "languages": languages,
            "modalities": modalities,
            "file_path": file_path,
            "total_samples": len(records),
            "filtered_count": total_filtered,
            "class_distribution": class_dist,
            "offset": offset,
            "limit": limit,
            "sample_slice": paged_samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset samples: {str(e)}")

