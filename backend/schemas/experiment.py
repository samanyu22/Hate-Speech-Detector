from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ExperimentRunRequest(BaseModel):
    exp_code: str = Field(..., description="exp1 to exp9")
    parameters: Optional[Dict[str, Any]] = None

class ExperimentResultResponse(BaseModel):
    id: str
    exp_code: str
    title: str
    status: str
    metrics: Dict[str, Any]
    visualizations: List[Dict[str, Any]] # e.g. confusion matrix data, bar charts, roc curve points
    summary: str
    created_at: str

