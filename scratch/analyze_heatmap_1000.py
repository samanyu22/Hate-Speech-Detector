import json
import time
from pathlib import Path
import numpy as np
from src.data.adapter import UniversalDatasetAdapter
from src.data.splitter import DatasetSplitter
from src.models.transformer_classifier import MultilingualTransformerEngine
from src.explainability.text_attribution import TextAttributionEngine

engine = MultilingualTransformerEngine()
engine.load_checkpoint()
xai = TextAttributionEngine()

# Load all 1260 academic samples
hasoc_path = "data/raw/hasoc_hi_en.jsonl"
hx_path = "data/raw/hatexplain_benchmark.jsonl"
subtle_path = "data/raw/subtle_structural_benchmark.jsonl"

records = UniversalDatasetAdapter(hasoc_path, text_column="text", label_column="label").load()
if Path(hx_path).exists():
    records += UniversalDatasetAdapter(hx_path, text_column="post_tokens", label_column="label").load()
if Path(subtle_path).exists():
    records += UniversalDatasetAdapter(subtle_path, text_column="text", label_column="label").load()

print(f"Total academic samples loaded: {len(records)}")

# Run comprehensive XAI audit across the entire test set and benchmark slice (1000 samples)
target_records = records[:1000]

start_time = time.time()
results = {
    "total_evaluated": len(target_records),
    "success_count": 0,
    "error_count": 0,
    "all_zero_count": 0,
    "faithful_count": 0,
    "faithfulness_scores": [],
    "tokens_evaluated": 0,
    "avg_tokens_per_sample": 0,
    "samples_with_active_rationales": 0
}

sample_breakdowns = []

for idx, r in enumerate(target_records):
    try:
        res = xai.compute_perturbation_attribution(r.text, engine.predict)
        results["success_count"] += 1
        attrs = res.get("token_attributions", [])
        faith = res.get("faithfulness_score", 0.0)
        results["faithfulness_scores"].append(faith)
        results["tokens_evaluated"] += len(attrs)

        scores = [a["score"] for a in attrs]
        non_zero = [s for s in scores if abs(s) >= 0.0001]

        if len(non_zero) == 0 and len(attrs) > 0:
            results["all_zero_count"] += 1
        else:
            results["samples_with_active_rationales"] += 1

        if faith >= 0.0:
            results["faithful_count"] += 1

    except Exception as e:
        results["error_count"] += 1

elapsed = time.time() - start_time
results["elapsed_seconds"] = round(elapsed, 2)
results["avg_faithfulness"] = round(float(np.mean(results["faithfulness_scores"])), 4)
results["median_faithfulness"] = round(float(np.median(results["faithfulness_scores"])), 4)
results["avg_tokens_per_sample"] = round(results["tokens_evaluated"] / max(1, results["total_evaluated"]), 1)

print("\n=== 1000 SAMPLE AUDIT RESULTS ===")
print(f"Total Evaluated: {results['total_evaluated']}")
print(f"Execution Success Rate: {(results['success_count'] / results['total_evaluated']) * 100:.2f}%")
print(f"Samples with Zero Attribution (Collapse): {results['all_zero_count']} ({(results['all_zero_count'] / results['total_evaluated']) * 100:.2f}%)")
print(f"Samples with Active Informative Tokens: {results['samples_with_active_rationales']} ({(results['samples_with_active_rationales'] / results['total_evaluated']) * 100:.2f}%)")
print(f"Average AOPC Faithfulness Score: {results['avg_faithfulness'] * 100:.2f}%")
print(f"Median Faithfulness Score: {results['median_faithfulness'] * 100:.2f}%")
print(f"Total Tokens Analyzed: {results['tokens_evaluated']}")
print(f"Time Taken: {results['elapsed_seconds']}s")

with open("scratch/heatmap_1000_audit.json", "w") as f:
    json.dump(results, f, indent=2)
