import json
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from backend.core.config import settings
from backend.database.db import get_connection
from src.experiments.runner import ExperimentRunner

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])
_runner: ExperimentRunner = None

def get_runner() -> ExperimentRunner:
    global _runner
    if _runner is None:
        _runner = ExperimentRunner()
    return _runner

@router.get("/")
def list_experiments():
    exp_defs = settings.experiments_config.get("experiments", {})
    conn = get_connection()
    completed = conn.execute("SELECT exp_code, title, created_at FROM experiments ORDER BY created_at DESC").fetchall()
    conn.close()

    status_map = {}
    for r in completed:
        if r["exp_code"] not in status_map:
            status_map[r["exp_code"]] = "completed"

    items = []
    for code, info in exp_defs.items():
        items.append({
            "code": info.get("id", code),
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "status": status_map.get(info.get("id", code), "pending")
        })
    return items

@router.post("/run/{exp_code}")
def run_experiment(exp_code: str):
    try:
        runner = get_runner()
        result = runner.run_experiment(exp_code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

