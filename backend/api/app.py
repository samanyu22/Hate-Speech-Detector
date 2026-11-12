"""
FastAPI Backend Application
Exposes RESTful endpoints for multimodal inference, dataset management,
model training, experiment execution, explainability, and settings.
"""

import os
import io
import json
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import yaml

from src.data.dataset_manager import DatasetManager
from src.classification.detector import MultimodalVerificationSystem
from src.experiments.runner import MasterExperimentRunner
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.explainability.text_explainer import TextExplainer
from src.explainability.visual_explainer import VisualExplainer

# Initialize FastAPI
app = FastAPI(
    title="Multimodal NLP Deepfake Detection API",
    description="CSET 346 Research System for Deepfake Detection & Explainable Content Verification",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
CONFIG_PATH = "config/app.yaml"
dm = DatasetManager()
detector = MultimodalVerificationSystem()
runner = MasterExperimentRunner(dataset_id="benchmark_300")
text_explainer = TextExplainer(detector.text_baseline)
visual_explainer = VisualExplainer(detector.vision_baseline)

# Asynchronous job tracking
background_jobs: Dict[str, Dict[str, Any]] = {}


def load_app_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def save_app_config(cfg: Dict[str, Any]):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


@app.get("/api/status")
def get_system_status():
    """Returns hardware specs, dataset status, and active models."""
    datasets = dm.list_datasets()
    active_dataset = next((d for d in datasets if d["dataset_id"] == "benchmark_300"), None)
    
    return {
        "status": "operational",
        "project": "CSET 346 Multimodal NLP for Deepfake Detection",
        "instructor": "Dr. Sakshi",
        "hardware": {
            "device": "CPU",
            "cores": os.cpu_count(),
            "framework": "PyTorch 2.13.0+cpu"
        },
        "dataset_summary": {
            "registered_datasets_count": len(datasets),
            "active_dataset": active_dataset["name"] if active_dataset else "None",
            "total_samples": active_dataset["total_samples"] if active_dataset else 0,
            "classes": active_dataset["classes"] if active_dataset else {}
        },
        "models_status": {
            "text_baseline_trained": detector.text_baseline.is_trained,
            "vision_baseline_trained": detector.vision_baseline.is_trained,
            "audio_baseline_trained": detector.audio_baseline.is_trained,
            "feature_fusion_ready": os.path.exists("models/checkpoints/multimodal_feature_fusion.pt"),
            "cross_attention_ready": os.path.exists("models/checkpoints/multimodal_cross_attention.pt")
        }
    }


@app.post("/api/predict")
async def verify_content(
    text: Optional[str] = Form(None),
    fusion_strategy: str = Form("cross_attention"),
    image_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None)
):
    """Processes multimodal inputs and generates 10 mandatory verification outputs."""
    img_bytes = await image_file.read() if image_file else None
    aud_bytes = await audio_file.read() if audio_file else None

    if not text and not img_bytes and not aud_bytes:
        raise HTTPException(status_code=400, detail="At least one modality (text, image, or audio) must be provided.")

    try:
        res = detector.verify_sample(
            text=text,
            image_source=img_bytes,
            audio_source=aud_bytes,
            fusion_strategy=fusion_strategy
        )

        # Add visual explainability heatmap if image was provided
        if img_bytes:
            try:
                vis_expl = visual_explainer.explain(img_bytes)
                res["visual_evidence"]["heatmap_b64"] = vis_expl["heatmap_b64"]
                res["visual_evidence"]["primary_focus"] = vis_expl["primary_focus_region"]
            except Exception:
                pass

        # Add text explainability token attributions if text was provided
        if text:
            try:
                t_expl = text_explainer.explain(text)
                res["textual_evidence"]["attributions"] = t_expl["attributions"]
                res["textual_evidence"]["top_evidence_tokens"] = t_expl["top_evidence_tokens"]
            except Exception:
                pass

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/datasets")
def list_datasets():
    return dm.list_datasets()


@app.get("/api/datasets/{dataset_id}/validate")
def validate_dataset(dataset_id: str):
    try:
        return dm.validate_dataset(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/datasets/register")
def register_dataset(
    dataset_id: str = Form(...),
    name: str = Form(...),
    manifest_path: str = Form(...),
    description: str = Form(""),
    source: str = Form("User Upload"),
    license: str = Form("Research Only")
):
    try:
        return dm.register_dataset(
            dataset_id=dataset_id,
            name=name,
            manifest_path=manifest_path,
            description=description,
            source=source,
            license=license
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/preprocess/inspect")
async def inspect_preprocessing(
    text: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None)
):
    """Returns detailed stage-by-stage preprocessing representations for UI inspection."""
    out = {}
    if text:
        tp = TextPreprocessor()
        out["text"] = tp.encode(text)
        out["text"]["input_ids"] = out["text"]["input_ids"].tolist()
        out["text"]["attention_mask"] = out["text"]["attention_mask"].tolist()

    if image_file:
        ip = ImagePreprocessor()
        img_bytes = await image_file.read()
        res = ip.preprocess(img_bytes)
        res.pop("tensor", None)  # Exclude raw tensor for clean JSON
        out["image"] = res

    if audio_file:
        ap = AudioPreprocessor()
        aud_bytes = await audio_file.read()
        res = ap.preprocess(aud_bytes)
        res.pop("tensor", None)
        out["audio"] = res

    return out


@app.get("/api/experiments")
def get_experiments():
    import math
    res = runner.load_cached_results()
    def clean(obj):
        if isinstance(obj, float):
            return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
        elif isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    return clean(res)


def _async_run_experiment(job_id: str, exp_num: int):
    try:
        background_jobs[job_id]["status"] = "running"
        if exp_num == 0:
            res = runner.run_all_experiments()
        else:
            res = runner.run_experiment(exp_num)
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["result"] = res
    except Exception as e:
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["error"] = str(e)


@app.post("/api/experiments/run/{exp_num}")
def trigger_experiment(exp_num: int, background_tasks: BackgroundTasks):
    job_id = f"job_exp_{exp_num}_{int(time.time())}"
    background_jobs[job_id] = {
        "job_id": job_id,
        "exp_num": exp_num,
        "status": "queued",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    background_tasks.add_task(_async_run_experiment, job_id, exp_num)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def check_job_status(job_id: str):
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return background_jobs[job_id]


@app.get("/api/settings")
def get_settings():
    return load_app_config()


@app.post("/api/settings")
def update_settings(settings: Dict[str, Any]):
    save_app_config(settings)
    return {"status": "updated", "settings": settings}


# Mount Frontend
frontend_dir = os.path.abspath("frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

