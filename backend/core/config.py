import os
from pathlib import Path
from typing import Any, Dict, Union, Optional
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"

class Settings:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.config_dir = CONFIG_DIR
        self.app_config = self._load_yaml(CONFIG_DIR / "app.yaml")
        self.datasets_config = self._load_yaml(CONFIG_DIR / "datasets" / "default.yaml")
        self.models_config = self._load_yaml(CONFIG_DIR / "models" / "default.yaml")
        self.languages_config = self._load_yaml(CONFIG_DIR / "languages" / "default.yaml")
        self.preprocessing_config = self._load_yaml(CONFIG_DIR / "preprocessing" / "default.yaml")
        self.rewards_config = self._load_yaml(CONFIG_DIR / "rewards" / "default.yaml")
        self.experiments_config = self._load_yaml(CONFIG_DIR / "experiments" / "default.yaml")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def resolve_path(self, path: Union[str, Path, None]) -> Path:
        """Dynamically resolve any relative or legacy absolute path against project BASE_DIR."""
        if not path:
            return self.base_dir
        p = Path(path)
        if p.exists():
            return p

        # Check if relative to BASE_DIR
        candidate = (self.base_dir / p).resolve()
        if candidate.exists():
            return candidate

        # If it was an absolute path from another machine (e.g., C:\Users\...\data\raw\...), extract project relative path
        norm = str(path).replace("\\", "/")
        for marker in ["data/", "models/", "config/", "reports/", "experiments/", "frontend/", "presentation/", "scratch/"]:
            if marker in norm:
                rel = norm[norm.find(marker):]
                c2 = (self.base_dir / rel).resolve()
                if c2.exists():
                    return c2

        # Check inside common directories by filename
        for sub in ["data/raw", "data/raw/images", "models/checkpoints", "models/versions"]:
            c3 = (self.base_dir / sub / p.name).resolve()
            if c3.exists():
                return c3

        # Default fallback
        return candidate

    def to_relative_path(self, path: Union[str, Path, None]) -> str:
        """Convert an absolute or machine-specific path to a portable relative POSIX path."""
        if not path:
            return ""
        norm = str(path).replace("\\", "/")
        base_norm = str(self.base_dir).replace("\\", "/")
        if norm.startswith(base_norm):
            return norm[len(base_norm):].lstrip("/")

        for marker in ["data/", "models/", "config/", "reports/", "experiments/", "frontend/", "presentation/", "scratch/"]:
            if marker in norm:
                return norm[norm.find(marker):]

        return norm

    @property
    def database_path(self) -> Path:
        rel = self.app_config.get("storage", {}).get("database_path", "data/hate_speech_detector.db")
        path = self.resolve_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def device(self) -> str:
        dev = os.getenv("DEVICE", self.app_config.get("compute", {}).get("device", "cpu"))
        return dev

    @property
    def reward_weights(self) -> Dict[str, float]:
        return self.rewards_config.get("rewards", {}).get("function", {}).get("weights", {
            "alpha": 0.35,
            "beta": 0.25,
            "gamma": 0.15,
            "delta": 0.25,
            "lambda_err": 0.50
        })

settings = Settings()

