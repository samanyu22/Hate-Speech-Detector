from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class OptimizationRequest(BaseModel):
    base_model_id: str = "multilingual-transformer-v1"
    method: str = Field(default="DPO", description="DPO or Preference-SFT")
    feedback_ids: Optional[List[str]] = None
    reward_weights: Optional[Dict[str, float]] = None
    learning_rate: float = 1.0e-5
    epochs: int = 2

class OptimizationResponse(BaseModel):
    run_id: str
    base_version: str
    new_version: str
    method: str
    status: str
    samples_trained: int
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    reward_gain: float
    promoted_to_active: bool
    message: str

