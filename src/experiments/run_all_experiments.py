import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import ExperimentRunner

def run_all():
    runner = ExperimentRunner()
    experiments = [f"exp{i}" for i in range(1, 10)]
    for exp in experiments:
        print(f"Executing {exp}...")
        res = runner.run_experiment(exp)
        print(f"Completed {exp}: {res['title']}")

if __name__ == "__main__":
    run_all()

