from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DatasetRegisterRequest(BaseModel):
    id: str
    name: str
    source: Optional[str] = "Uploaded Research Dataset"
    license: Optional[str] = "Research Use"
    file_path: str
    modalities: List[str] = ["text"]
    languages: List[str] = ["en"]
    text_column: str = "text"
    label_column: str = "label"
    image_column: Optional[str] = None
    context_column: Optional[str] = None
    label_mapping: Dict[str, int] = {"0": 0, "1": 1}

class DatasetValidationResponse(BaseModel):
    is_valid: bool
    sample_count: int
    class_distribution: Dict[str, int]
    missing_text_count: int
    missing_image_count: int
    duplicate_count: int
    detected_languages: Dict[str, int]
    warnings: List[str]
    report_summary: str

