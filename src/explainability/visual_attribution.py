from typing import List, Dict, Any, Optional

class VisualAttributionEngine:
    @staticmethod
    def map_ocr_attributions(ocr_boxes: List[Dict[str, Any]], token_attributions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Maps textual token attribution scores back to image OCR bounding regions.
        """
        if not ocr_boxes or not token_attributions:
            return []

        # Create quick lookup for word scores
        token_score_map = {a["token"].lower(): a["score"] for a in token_attributions}
        visual_regions = []

        for b in ocr_boxes:
            box_text = b.get("text", "")
            box_words = box_text.lower().split()
            # Average score of words inside this box
            scores = [token_score_map.get(w, 0.0) for w in box_words]
            avg_score = sum(scores) / max(1, len(scores))

            visual_regions.append({
                "box": b.get("box", [0, 0, 100, 100]),
                "region_label": box_text,
                "attribution_score": round(float(avg_score), 4),
                "ocr_confidence": b.get("confidence", 1.0)
            })

        return visual_regions

