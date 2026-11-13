from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

class OCRService:
    _reader = None

    @classmethod
    def get_reader(cls):
        if cls._reader is None:
            try:
                import easyocr
                # Initialize English and Hindi OCR
                cls._reader = easyocr.Reader(["en", "hi"], gpu=False)
            except Exception as e:
                print(f"Warning: easyocr initialization notice: {e}")
                cls._reader = None
        return cls._reader

    @classmethod
    def extract_text(cls, image_input: Any) -> Dict[str, Any]:
        reader = cls.get_reader()
        if reader is None:
            return {
                "extracted_text": "",
                "confidence": 0.0,
                "bounding_boxes": [],
                "engine": "unavailable"
            }

        try:
            # easyocr accepts filepath, numpy array, or PIL image
            if isinstance(image_input, (str, Path)):
                img_path = str(image_input)
                results = reader.readtext(img_path)
            elif isinstance(image_input, Image.Image):
                np_img = np.array(image_input.convert("RGB"))
                results = reader.readtext(np_img)
            else:
                np_img = np.array(image_input)
                results = reader.readtext(np_img)

            extracted_chunks = []
            boxes = []
            confidences = []

            for bbox, text, conf in results:
                extracted_chunks.append(text.strip())
                confidences.append(float(conf))
                # bbox is list of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                ymin = int(min(pt[1] for pt in bbox))
                xmin = int(min(pt[0] for pt in bbox))
                ymax = int(max(pt[1] for pt in bbox))
                xmax = int(max(pt[0] for pt in bbox))
                boxes.append({
                    "box": [ymin, xmin, ymax, xmax],
                    "text": text.strip(),
                    "confidence": round(float(conf), 3)
                })

            full_text = " ".join(extracted_chunks)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            return {
                "extracted_text": full_text,
                "confidence": round(avg_conf, 3),
                "bounding_boxes": boxes,
                "engine": "easyocr"
            }
        except Exception as e:
            return {
                "extracted_text": "",
                "confidence": 0.0,
                "bounding_boxes": [],
                "error": str(e),
                "engine": "easyocr_failed"
            }

