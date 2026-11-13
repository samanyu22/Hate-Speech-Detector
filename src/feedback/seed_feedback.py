import uuid
import json
from datetime import datetime
from backend.database.db import get_connection
from src.rewards.reward_function import MultiObjectiveRewardCalculator

def seed_sample_feedback():
    conn = get_connection()
    cursor = conn.cursor()
    calculator = MultiObjectiveRewardCalculator()

    seed_items = [
        {
            "text": "Maine suna ek neta keh raha tha ki 'inko goli maaro', aisi nafrat ka hum virodh karte hain.",
            "pred_label": 1, # Model initially flagged due to 'goli maaro' keyword
            "conf": 0.82,
            "is_correct": False,
            "suggested": 0,
            "rating": 2,
            "flags": ["sarcasm_counter", "quotation"],
            "comment": "The user is explicitly protesting hate speech ('hum virodh karte hain'), not promoting it."
        },
        {
            "text": "Bhai kal ka cricket match dekh ke maza aa gaya, kya mast sixes maare!",
            "pred_label": 0,
            "conf": 0.94,
            "is_correct": True,
            "suggested": None,
            "rating": 5,
            "flags": [],
            "comment": "Accurate non-hateful classification for sports Hinglish."
        },
        {
            "text": "Ye saale kutto jaise bhauk rahe hain, inko raste me goli maar do sabhi ko.",
            "pred_label": 1,
            "conf": 0.96,
            "is_correct": True,
            "suggested": None,
            "rating": 5,
            "flags": ["threat", "dehumanization"],
            "comment": "Correctly identified violent threat and animalistic dehumanization."
        },
        {
            "text": "Oh sure, blame every single problem in the world on immigrants, brilliant logic!",
            "pred_label": 1, # Misclassified sarcasm as genuine xenophobia
            "conf": 0.74,
            "is_correct": False,
            "suggested": 0,
            "rating": 2,
            "flags": ["sarcasm"],
            "comment": "Sarcastic rebuttal ridiculing anti-immigrant scapegoating."
        },
        {
            "text": "Disagreements during policy debates are healthy for any functioning democracy.",
            "pred_label": 0,
            "conf": 0.91,
            "is_correct": True,
            "suggested": None,
            "rating": 5,
            "flags": [],
            "comment": "Correctly preserved civil political discourse."
        },
        {
            "text": "Tumhari kaum hi aisi hai haramkhor, tum sabko genocide karke khatam kar dena chahiye.",
            "pred_label": 1,
            "conf": 0.98,
            "is_correct": True,
            "suggested": None,
            "rating": 5,
            "flags": ["genocide_incitement"],
            "comment": "Genuinely severe hate speech accurately flagged."
        }
    ]

    for item in seed_items:
        pred_id = str(uuid.uuid4())
        fb_id = str(uuid.uuid4())
        rew_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Insert prediction
        cursor.execute("""
            INSERT INTO predictions
            (id, input_text, predicted_label, confidence, model_id, model_version, language_detected, modalities_used, explanation_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred_id,
            item["text"],
            item["pred_label"],
            item["conf"],
            "multilingual-transformer-v1",
            "1.0",
            "hi-en" if " " in item["text"] else "en",
            json.dumps(["text"]),
            json.dumps({"summary": "Attribution computed"}),
            now
        ))

        # Insert feedback
        cursor.execute("""
            INSERT INTO feedback
            (id, prediction_id, is_correct, suggested_label, explanation_rating, flags, comment, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fb_id,
            pred_id,
            1 if item["is_correct"] else 0,
            item["suggested"],
            item["rating"],
            json.dumps(item["flags"]),
            item["comment"],
            "accepted",
            now
        ))

        # Compute and insert reward
        rew = calculator.calculate_reward(
            predicted_label=item["pred_label"],
            confidence=item["conf"],
            is_correct=item["is_correct"],
            suggested_label=item["suggested"],
            explanation_rating=item["rating"],
            context_used=False
        )

        cursor.execute("""
            INSERT INTO rewards
            (id, prediction_id, feedback_id, r_classification, r_explanation, r_context, r_feedback, r_error, total_reward, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rew_id,
            pred_id,
            fb_id,
            rew["r_classification"],
            rew["r_explanation"],
            rew["r_context"],
            rew["r_feedback"],
            rew["r_error"],
            rew["total_reward"],
            json.dumps(rew),
            now
        ))

    conn.commit()
    conn.close()
    print(f"Seeded {len(seed_items)} realistic feedback and reward records.")

if __name__ == "__main__":
    seed_sample_feedback()

