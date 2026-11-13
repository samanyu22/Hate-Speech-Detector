from typing import List, Dict, Any
from backend.database.db import get_connection

class PreferenceDatasetBuilder:
    @staticmethod
    def build_preference_pairs() -> List[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        rows = cursor.execute("""
            SELECT f.id as feedback_id, f.suggested_label, f.is_correct,
                   p.input_text, p.predicted_label, p.confidence, p.id as prediction_id
            FROM feedback f
            JOIN predictions p ON f.prediction_id = p.id
            WHERE f.status = 'accepted'
        """).fetchall()
        conn.close()

        pairs = []
        for r in rows:
            text = r["input_text"]
            pred = r["predicted_label"]
            is_corr = bool(r["is_correct"])
            sugg = r["suggested_label"]

            if not is_corr and sugg is not None:
                # The corrected label is the winner, the incorrect prediction is the loser
                pairs.append({
                    "text": text,
                    "winning_label": int(sugg),
                    "losing_label": int(pred),
                    "confidence_margin": float(r["confidence"]),
                    "source": "human_correction"
                })
            elif is_corr:
                # Prediction was confirmed correct
                pairs.append({
                    "text": text,
                    "winning_label": int(pred),
                    "losing_label": 1 - int(pred),
                    "confidence_margin": float(r["confidence"]),
                    "source": "human_confirmation"
                })

        return pairs

