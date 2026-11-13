import json
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.schemas.optimization import OptimizationRequest, OptimizationResponse
from backend.database.db import get_connection
from src.optimization.optimizer import RewardOptimizer
from backend.workers.task_manager import task_manager

router = APIRouter(prefix="/api/optimization", tags=["Reward Optimization"])

@router.post("/run", response_model=OptimizationResponse)
def trigger_optimization(req: OptimizationRequest):
    try:
        optimizer = RewardOptimizer(base_model_id=req.base_model_id)
        result = optimizer.run_optimization_cycle(
            method=req.method,
            lr=req.learning_rate,
            epochs=req.epochs
        )
        if result.get("status") == "skipped":
            raise HTTPException(status_code=400, detail=result.get("message"))

        return OptimizationResponse(
            run_id=result["run_id"],
            base_version=result["base_version"],
            new_version=result["new_version"],
            method=result["method"],
            status=result["status"],
            samples_trained=result["samples_trained"],
            before_metrics=result["before_metrics"],
            after_metrics=result["after_metrics"],
            reward_gain=result["reward_gain"],
            promoted_to_active=result["promoted_to_active"],
            message=result["message"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
def get_optimization_history():
    conn = get_connection()
    runs = conn.execute("SELECT * FROM optimization_runs ORDER BY created_at DESC").fetchall()
    versions = conn.execute("SELECT * FROM model_versions ORDER BY created_at DESC").fetchall()
    rewards = conn.execute("SELECT * FROM rewards ORDER BY created_at DESC LIMIT 50").fetchall()
    conn.close()

    run_list = []
    for r in runs:
        item = dict(r)
        if item.get("loss_history"):
            item["loss_history"] = json.loads(item["loss_history"])
        if item.get("metrics_before"):
            item["metrics_before"] = json.loads(item["metrics_before"])
        if item.get("metrics_after"):
            item["metrics_after"] = json.loads(item["metrics_after"])
        run_list.append(item)

    reward_list = []
    for rew in rewards:
        item = dict(rew)
        if item.get("details"):
            item["details"] = json.loads(item["details"])
        reward_list.append(item)

    return {
        "runs": run_list,
        "versions": [dict(v) for v in versions],
        "recent_rewards": reward_list
    }

