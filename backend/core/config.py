import os
from pathlib import Path
from typing import Any, Dict
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

    @property
    def database_path(self) -> Path:
        rel = self.app_config.get("storage", {}).get("database_path", "data/hate_speech_detector.db")
        path = self.base_dir / rel
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

