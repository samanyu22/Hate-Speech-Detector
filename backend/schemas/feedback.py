from pydantic import BaseModel, Field
from typing import List, Optional

class FeedbackCreate(BaseModel):
    prediction_id: str
    is_correct: bool = Field(..., description="Whether model prediction was accurate")
    suggested_label: Optional[int] = Field(default=None, description="0=Non-hateful, 1=Hateful if user suggests correction")
    explanation_rating: Optional[int] = Field(default=None, ge=1, le=5, description="1 to 5 rating of explanation")
    flags: Optional[List[str]] = Field(default_factory=list, description="Tags: e.g. 'context_misunderstood', 'sarcasm'")
    comment: Optional[str] = Field(default=None, description="Detailed user feedback")

class FeedbackResponse(BaseModel):
    id: str
    prediction_id: str
    status: str
    created_at: str

