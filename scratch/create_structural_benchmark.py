import json
import csv
import io
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scratch.test_user_dataset import raw_tsv

reader = csv.DictReader(io.StringIO(raw_tsv.strip()), delimiter="\t")
records = []
for row in reader:
    sample_id = row["ID"]
    text = row["Text"]
    label_str = row["Label"].strip().lower()
    label = 1 if label_str == "hateful" else 0
    records.append({
        "id": sample_id,
        "text": text,
        "label": label,
        "language": "en",
        "category": "structural_discrimination" if label == 1 else "civil_counter_speech"
    })

out_path = "data/raw/subtle_structural_benchmark.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Successfully saved {len(records)} records to {out_path}!")
