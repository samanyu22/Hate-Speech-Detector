import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from backend.core.config import settings
from backend.database.db import get_connection

router = APIRouter(prefix="/api/reports", tags=["Reports & Research"])

@router.get("/summary")
def get_research_summary():
    conn = get_connection()
    c = conn.cursor()

    # 1. Registered Datasets
    raw_datasets = c.execute("""
        SELECT id, name, source, license, languages, modalities, file_path, sample_count, class_distribution
        FROM datasets
        ORDER BY sample_count DESC
    """).fetchall()

    datasets_list = []
    total_samples = 0
    total_hateful_in_datasets = 0
    total_nonhateful_in_datasets = 0

    for r in raw_datasets:
        d = dict(r)
        d["languages"] = json.loads(d["languages"]) if d.get("languages") else []
        d["modalities"] = json.loads(d["modalities"]) if d.get("modalities") else []
        dist = json.loads(d["class_distribution"]) if d.get("class_distribution") else {}
        d["class_distribution"] = dist
        s_count = d.get("sample_count", 0)
        total_samples += s_count
        total_hateful_in_datasets += dist.get("1", dist.get(1, 0))
        total_nonhateful_in_datasets += dist.get("0", dist.get(0, 0))
        datasets_list.append(d)

    # 2. Model Registry & Active Model
    model_count = c.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0]
    active_m = c.execute("SELECT * FROM model_registry WHERE is_active = 1").fetchone()
    active_model = dict(active_m) if active_m else {}
    if active_model.get("metrics"):
        try:
            active_model["metrics"] = json.loads(active_model["metrics"])
        except Exception:
            pass

    # 3. Live Predictions Telemetry
    total_preds = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    hateful_preds = c.execute("SELECT COUNT(*) FROM predictions WHERE predicted_label = 1").fetchone()[0]
    nonhateful_preds = c.execute("SELECT COUNT(*) FROM predictions WHERE predicted_label = 0").fetchone()[0]
    raw_langs = c.execute("SELECT language_detected, COUNT(*) FROM predictions GROUP BY language_detected").fetchall()
    lang_counts = {r[0]: r[1] for r in raw_langs if r[0]}

    # 4. Human Feedback Audited
    total_fb = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    correct_fb = c.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 1").fetchone()[0]
    incorrect_fb = c.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 0").fetchone()[0]
    agreement_rate = round((correct_fb / total_fb * 100), 1) if total_fb > 0 else 0.0

    # 5. Reward Optimization Cycles
    opt_count = c.execute("SELECT COUNT(*) FROM optimization_runs").fetchone()[0]
    latest_opt_row = c.execute("SELECT * FROM optimization_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    latest_opt = dict(latest_opt_row) if latest_opt_row else None
    if latest_opt and latest_opt.get("metrics_after"):
        try:
            latest_opt["metrics_after"] = json.loads(latest_opt["metrics_after"])
        except Exception:
            pass

    # 6. Research Experiments Suite (exp1 to exp9)
    distinct_completed = c.execute("SELECT COUNT(DISTINCT exp_code) FROM experiments WHERE status = 'completed'").fetchone()[0]
    total_exp_runs = c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    # 7. Recent Inferences Feed (latest 10 live predictions)
    recent_rows = c.execute("""
        SELECT p.id, p.input_text, p.predicted_label, p.confidence, p.language_detected, p.modalities_used, p.created_at,
               f.is_correct, f.suggested_label
        FROM predictions p
        LEFT JOIN feedback f ON p.id = f.prediction_id
        ORDER BY p.created_at DESC LIMIT 10
    """).fetchall()
    recent_preds = [dict(r) for r in recent_rows]

    conn.close()

    return {
        "title": "Hate Speech Detection through Explainable AI and Reward Optimized LLMs",
        "supervisor": "Dr. Pradeepika Verma",
        "total_datasets": len(datasets_list),
        "total_samples": total_samples,
        "datasets": datasets_list,
        "dataset_class_totals": {
            "hateful": total_hateful_in_datasets,
            "non_hateful": total_nonhateful_in_datasets
        },
        "total_models": model_count,
        "active_model_name": active_model.get("name", "Multilingual Code-Mixed Transformer Classifier"),
        "active_model_version": active_model.get("version", "1.0"),
        "active_model": active_model,
        "total_predictions": total_preds,
        "predictions_summary": {
            "total": total_preds,
            "hateful": hateful_preds,
            "non_hateful": nonhateful_preds,
            "languages": lang_counts
        },
        "total_feedbacks": total_fb,
        "feedback_summary": {
            "total": total_fb,
            "correct": correct_fb,
            "incorrect": incorrect_fb,
            "agreement_rate": agreement_rate
        },
        "optimization_cycles": opt_count,
        "latest_optimization": latest_opt,
        "completed_experiments": distinct_completed,
        "distinct_completed_experiments": distinct_completed,
        "total_experiment_runs": total_exp_runs,
        "experiments_suite_total": 9,
        "recent_predictions": recent_preds,
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

