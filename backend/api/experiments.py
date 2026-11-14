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
    completed = conn.execute("SELECT exp_code, title, created_at, results_json FROM experiments ORDER BY created_at DESC").fetchall()
    conn.close()

    latest_by_code = {}
    for r in completed:
        code = r["exp_code"]
        if code not in latest_by_code:
            res_data = None
            if r["results_json"]:
                try:
                    res_data = json.loads(r["results_json"])
                except Exception:
                    pass
            latest_by_code[code] = {
                "created_at": r["created_at"],
                "results": res_data
            }

    items = []
    for code, info in exp_defs.items():
        exp_id = info.get("id", code)
        run_info = latest_by_code.get(exp_id)
        items.append({
            "code": exp_id,
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "status": "completed" if run_info else "pending",
            "last_executed": run_info["created_at"] if run_info else None,
            "results": run_info["results"] if run_info else None
        })
    return items

@router.post("/reset")
def reset_experiments():
    try:
        conn = get_connection()
        conn.execute("DELETE FROM experiments")
        conn.commit()
        conn.close()
        return {"status": "success", "message": "All experiment records reset to 0/9"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run-all")
def run_all_experiments():
    try:
        runner = get_runner()
        results = {}
        for code in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9"]:
            results[code] = runner.run_experiment(code)
        return {"status": "success", "total_executed": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run/{exp_code}")
def run_experiment(exp_code: str):
    try:
        runner = get_runner()
        result = runner.run_experiment(exp_code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

