import requests
import time

print("Waiting for server...")
for _ in range(10):
    try:
        r = requests.get("http://127.0.0.1:8000/api/health")
        if r.status_code == 200:
            print("Server is up and healthy!")
            break
    except Exception:
        time.sleep(1)

test_inputs = [
    "nigger",
    "I hate niggers",
    "You are a nigger",
    "sniggering at the joke",
    "chink",
    "faggot",
    "People from that religion are inferior",
    "Our school welcomes students from every religious background"
]

print("\n=== LIVE API TESTS ON NEWLY RESTARTED SERVER ===")
for text in test_inputs:
    resp = requests.post("http://127.0.0.1:8000/api/detection/detect", json={"text": text})
    data = resp.json()
    print(f"'{text}' -> {data.get('label_name')} (label={data.get('predicted_label')}, conf={data.get('confidence')})")
