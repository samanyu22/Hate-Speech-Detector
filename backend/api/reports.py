from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from backend.core.config import settings
from backend.database.db import get_connection

router = APIRouter(prefix="/api/reports", tags=["Reports & Research"])

@router.get("/summary")
def get_research_summary():
    conn = get_connection()
    dataset_count = conn.execute("SELECT COUNT(*), SUM(sample_count) FROM datasets").fetchone()
    model_count = conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()
    pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
    feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()
    opt_count = conn.execute("SELECT COUNT(*) FROM optimization_runs").fetchone()
    exp_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()

    active_model = conn.execute("SELECT * FROM model_registry WHERE is_active = 1").fetchone()
    conn.close()

    return {
        "title": "Hate Speech Detection through Explainable AI and Reward Optimized LLMs",
        "supervisor": "Dr. Pradeepika Verma",
        "total_datasets": dataset_count[0] if dataset_count else 0,
        "total_samples": dataset_count[1] if dataset_count else 0,
        "total_models": model_count[0] if model_count else 0,
        "active_model_name": active_model["name"] if active_model else "Multilingual Transformer",
        "active_model_version": active_model["version"] if active_model else "1.0",
        "total_predictions": pred_count[0] if pred_count else 0,
        "total_feedbacks": feedback_count[0] if feedback_count else 0,
        "optimization_cycles": opt_count[0] if opt_count else 0,
        "completed_experiments": exp_count[0] if exp_count else 0,
        "modalities": ["Text", "Visual/Memes", "EasyOCR", "Contextual Dialogue"],
        "languages": ["English", "Hindi (Devanagari)", "Hindi-English Code-Mixed (Hinglish)"]
    }

@router.get("/paper-draft")
def get_paper_draft():
    paper_path = settings.base_dir / "paper" / "main.tex"
    if paper_path.exists():
        with open(paper_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"tex_content": content, "format": "ACL LaTeX"}
    return {"tex_content": "Drafting in progress...", "format": "ACL LaTeX"}

