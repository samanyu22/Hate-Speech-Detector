import json
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from backend.database.db import get_connection

router = APIRouter(prefix="/api/models", tags=["Model Registry"])

@router.get("/")
def list_models():
    conn = get_connection()
    models = conn.execute("SELECT * FROM model_registry ORDER BY created_at DESC").fetchall()
    versions = conn.execute("SELECT * FROM model_versions ORDER BY created_at DESC").fetchall()
    conn.close()

    model_list = []
    for m in models:
        item = dict(m)
        if item.get("metrics"):
            item["metrics"] = json.loads(item["metrics"])
        model_list.append(item)

    version_list = []
    for v in versions:
        item = dict(v)
        if item.get("eval_details"):
            item["eval_details"] = json.loads(item["eval_details"])
        version_list.append(item)

    return {
        "models": model_list,
        "versions": version_list
    }

@router.post("/select/{model_id}")
def select_active_model(model_id: str):
    conn = get_connection()
    # Check model exists
    row = conn.execute("SELECT * FROM model_registry WHERE id = ?", (model_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Model ID not found")

    conn.execute("UPDATE model_registry SET is_active = 0")
    conn.execute("UPDATE model_registry SET is_active = 1 WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()

    # Trigger hot-reload in detection engine
    from backend.api.detection import get_models
    get_models(force_reload=True)

    return {"status": "success", "active_model": model_id}

