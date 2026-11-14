import requests

test_inputs = [
    "i hate jews",
    "I hate Muslims",
    "I hate Hindus",
    "I hate Dalits",
    "I hate women",
    "I hate black people",
    "kill all jews",
    "jews are parasites",
    "I hate pollution",
    "I hate Mondays"
]

print("=== LIVE SERVER TEST: DIRECT HATRED & CONTROL SAMPLES ===")
for text in test_inputs:
    resp = requests.post("http://127.0.0.1:8000/api/detection/detect", json={"text": text})
    data = resp.json()
    print(f"'{text}' -> {data.get('label_name')} (label={data.get('predicted_label')}, conf={data.get('confidence')})")

