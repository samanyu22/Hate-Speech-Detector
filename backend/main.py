import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to sys.path so `python backend/main.py` works directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.core.config import settings
from backend.database.db import init_db
from backend.api import detection, feedback, datasets, models, optimization, fairness, experiments, reports, settings as settings_api

from starlette.middleware.base import BaseHTTPMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database schema on startup
    init_db()
    yield

app = FastAPI(
    title="Hate Speech Detection through Explainable AI and Reward Optimized LLMs",
    description="Official Academic Research Platform: HS Detector (Explainable AI & Reward Optimized LLMs)",
    version="1.0.0",
    lifespan=lifespan
)

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

# Enable CORS for local research interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(detection.router)
app.include_router(feedback.router)
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(optimization.router)
app.include_router(fairness.router)
app.include_router(experiments.router)
app.include_router(reports.router)
app.include_router(settings_api.router)

# Mount Frontend Static Directories
frontend_dir = settings.base_dir / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    if (frontend_dir / "css").exists():
        app.mount("/css", StaticFiles(directory=str(frontend_dir / "css")), name="css")
    if (frontend_dir / "js").exists():
        app.mount("/js", StaticFiles(directory=str(frontend_dir / "js")), name="js")

presentation_dir = settings.base_dir / "presentation"
if presentation_dir.exists():
    app.mount("/presentation", StaticFiles(directory=str(presentation_dir)), name="presentation")

images_dir = settings.base_dir / "data" / "raw" / "images"
if images_dir.exists():
    app.mount("/data/raw/images", StaticFiles(directory=str(images_dir)), name="dataset_images")

@app.get("/")
def serve_index():
    index_file = settings.base_dir / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Hate Speech Detection API Online. Visit /docs for Swagger API."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app": "Hate Speech Detection through Explainable AI & Reward Optimized LLMs",
        "device": settings.device,
        "database": str(settings.database_path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
