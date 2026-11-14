import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.schemas.optimization import OptimizationRequest, OptimizationResponse
from backend.database.db import get_connection
from src.optimization.optimizer import RewardOptimizer
from src.rewards.reward_function import MultiObjectiveRewardCalculator
from backend.workers.task_manager import task_manager

router = APIRouter(prefix="/api/optimization", tags=["Reward Optimization"])

class RewardSimulationRequest(BaseModel):
    predicted_label: int = 1
    confidence: float = 0.95
    is_correct: Optional[bool] = True
    suggested_label: Optional[int] = None
    faithfulness_score: float = 0.78
    explanation_rating: Optional[int] = 5
    context_used: bool = True
    is_biased_false_positive: bool = False
    weights: Optional[Dict[str, float]] = None

@router.post("/simulate-reward")
def simulate_reward(req: RewardSimulationRequest):
    try:
        calc = MultiObjectiveRewardCalculator(weights=req.weights)
        res = calc.calculate_reward(
            predicted_label=req.predicted_label,
            confidence=req.confidence,
            is_correct=req.is_correct,
            suggested_label=req.suggested_label,
            faithfulness_score=req.faithfulness_score,
            explanation_rating=req.explanation_rating,
            context_used=req.context_used,
            is_biased_false_positive=req.is_biased_false_positive
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    total_rewards_count = conn.execute("SELECT COUNT(*) FROM rewards").fetchone()[0]
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

    mean_reward = 0.0
    if reward_list:
        tot_sum = sum(float(r.get("total_reward") or 0.0) for r in reward_list)
        mean_reward = round(tot_sum / len(reward_list), 3)

    return {
        "runs": run_list,
        "versions": [dict(v) for v in versions],
        "recent_rewards": reward_list,
        "summary": {
            "total_pairs": total_rewards_count,
            "active_method": "DPO (Direct Preference Optimization)",
            "mean_reward": mean_reward,
            "total_runs": len(run_list),
            "latest_run": run_list[0] if run_list else None
        }
    }

