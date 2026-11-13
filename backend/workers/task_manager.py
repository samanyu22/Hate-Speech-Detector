import uuid
import threading
import json
from datetime import datetime
from typing import Callable, Any, Dict, Optional
from backend.database.db import get_connection

class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def submit_task(self, name: str, fn: Callable, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        conn = get_connection()
        conn.execute(
            "INSERT INTO background_tasks (task_id, name, status, progress, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, name, "running", 0.0, now, now)
        )
        conn.commit()
        conn.close()

        def _worker():
            try:
                result = fn(*args, **kwargs)
                conn = get_connection()
                conn.execute(
                    "UPDATE background_tasks SET status = ?, progress = ?, result_json = ?, updated_at = ? WHERE task_id = ?",
                    ("completed", 1.0, json.dumps(result), datetime.utcnow().isoformat(), task_id)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                conn = get_connection()
                conn.execute(
                    "UPDATE background_tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",
                    ("failed", str(e), datetime.utcnow().isoformat(), task_id)
                )
                conn.commit()
                conn.close()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return task_id

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM background_tasks WHERE task_id = ?", (task_id,)).fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        if res.get("result_json"):
            try:
                res["result"] = json.loads(res["result_json"])
            except Exception:
                res["result"] = res["result_json"]
        return res

task_manager = TaskManager()

