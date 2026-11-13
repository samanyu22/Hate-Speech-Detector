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
def inspect_dataset(dataset_id: str, limit: int = 15):
    conn = get_connection()
    row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = row["file_path"]
    try:
        adapter = UniversalDatasetAdapter(file_path=file_path)
        records = adapter.load()
        samples = [r.to_dict() for r in records[:limit]]
        return {
            "dataset_id": dataset_id,
            "total_samples": len(records),
            "sample_slice": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset samples: {str(e)}")

