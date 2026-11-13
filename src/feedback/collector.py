import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from backend.database.db import get_connection

class FeedbackCollector:
    @staticmethod
    def submit_feedback(
        prediction_id: str,
        is_correct: bool,
        suggested_label: Optional[int] = None,
        explanation_rating: Optional[int] = None,
        flags: Optional[List[str]] = None,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        conn = get_connection()
        cursor = conn.cursor()

        # Check prediction exists
        pred_row = cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        if not pred_row:
            conn.close()
            raise ValueError(f"Prediction ID {prediction_id} not found.")

        # Quality Control: Validate consistency
        status = "accepted"
        pred_label = pred_row["predicted_label"]
        if is_correct and suggested_label is not None and suggested_label != pred_label:
            # Inconsistent: marked correct but suggested opposite label
            status = "rejected_inconsistent"

        feedback_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO feedback (id, prediction_id, is_correct, suggested_label, explanation_rating, flags, comment, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback_id,
            prediction_id,
            1 if is_correct else 0,
            suggested_label,
            explanation_rating,
            json.dumps(flags or []),
            comment,
            status
        ))
        conn.commit()
        conn.close()

        return {
            "feedback_id": feedback_id,
            "prediction_id": prediction_id,
            "status": status,
            "recorded_at": datetime.now(timezone.utc).isoformat()
        }

    @staticmethod
    def get_eligible_feedback() -> List[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        rows = cursor.execute("""
            SELECT f.*, p.input_text, p.predicted_label, p.confidence, p.model_version
            FROM feedback f
            JOIN predictions p ON f.prediction_id = p.id
            WHERE f.status = 'accepted'
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
