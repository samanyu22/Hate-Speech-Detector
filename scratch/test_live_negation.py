import requests
import time

tests = [
    "i hate jews",
    "i dont hate jews",
    "i do not hate jews",
    "I hate Muslims",
    "we never hate muslims",
    "I don't hate women",
    "nigger",
    "sniggering at the joke",
    "I hate pollution"
]

print("=== LIVE SERVER NEGATION API TEST ===")
for t in tests:
    try:
        r = requests.post("http://127.0.0.1:8000/api/detection/detect", json={"text": t})
        data = r.json()
        print(f"'{t}' -> {data.get('label_name')} (label={data.get('predicted_label')}, conf={data.get('confidence')})")
    except Exception as e:
        print(f"'{t}' -> ERROR: {e}")

