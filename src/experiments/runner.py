"""
Master Experiment Runner
Coordinates execution of all 6 required academic experiments,
stores structured evaluation metrics in persistent JSON storage, and tracks execution status.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional

from src.data.dataset_manager import DatasetManager
from src.classification.detector import MultimodalVerificationSystem
from src.experiments.exp1_baselines import Experiment1Runner
from src.experiments.exp2_fusion import Experiment2Runner
from src.experiments.exp3_generalization import Experiment3Runner
from src.experiments.exp4_ablation import Experiment4Runner
from src.experiments.exp5_error_analysis import Experiment5Runner
from src.experiments.exp6_explanation_quality import Experiment6Runner


class MasterExperimentRunner:
    """Orchestrates and caches the 6 required experiments."""

    def __init__(
        self,
        results_file: str = "reports/experiment_results.json",
        dataset_id: str = "benchmark_300"
    ):
        self.results_file = results_file
        self.dataset_id = dataset_id
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        
        self.dm = DatasetManager()
        self.detector = MultimodalVerificationSystem()
        
        self.exp1 = Experiment1Runner(self.detector, self.dm)
        self.exp2 = Experiment2Runner(self.detector, self.dm)
        self.exp3 = Experiment3Runner(self.detector, self.dm)
        self.exp4 = Experiment4Runner(self.detector, self.dm)
        self.exp5 = Experiment5Runner(self.detector, self.dm)
        self.exp6 = Experiment6Runner(self.detector, self.dm)

    def load_cached_results(self) -> Dict[str, Any]:
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_results(self, results: Dict[str, Any]):
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def run_experiment(self, exp_num: int) -> Dict[str, Any]:
        """Runs an individual experiment by number (1 to 6)."""
        cached = self.load_cached_results()
        start_t = time.time()

        if exp_num == 1:
            res = self.exp1.run(self.dataset_id)
        elif exp_num == 2:
            res = self.exp2.run(self.dataset_id)
        elif exp_num == 3:
            res = self.exp3.run(self.dataset_id)
        elif exp_num == 4:
            res = self.exp4.run(self.dataset_id)
        elif exp_num == 5:
            res = self.exp5.run(self.dataset_id)
        elif exp_num == 6:
            res = self.exp6.run(self.dataset_id)
        else:
            raise ValueError(f"Invalid experiment number: {exp_num}. Expected 1 to 6.")

        res["execution_time_sec"] = round(time.time() - start_t, 2)
        res["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        cached[f"experiment_{exp_num}"] = res
        self.save_results(cached)
        return res

    def run_all_experiments(self) -> Dict[str, Any]:
        """Runs all 6 experiments sequentially."""
        all_results = {}
        for i in range(1, 7):
            print(f"Executing Experiment {i}/6...")
            all_results[f"experiment_{i}"] = self.run_experiment(i)
        return all_results

