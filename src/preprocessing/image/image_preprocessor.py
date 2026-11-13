import io
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def load_image(self, source: str) -> Image.Image:
        # Check if source is base64 string
        if source.startswith("data:image") or len(source) > 500 and not os.path.exists(source):
            if "base64," in source:
                source = source.split("base64,")[1]
            img_bytes = base64.b64decode(source)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Source is file path
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {source}")
        return Image.open(path).convert("RGB")

    def inspect_and_preprocess(self, image_input: Any) -> Dict[str, Any]:
        if isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            img = self.load_image(str(image_input))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_width, orig_height = img.size
        # Resize to target
        resized = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img_np = np.array(resized).astype(np.float32) / 255.0

        # Calculate visual properties
        mean_brightness = float(np.mean(img_np))
        contrast = float(np.std(img_np))

        return {
            "original_resolution": [orig_width, orig_height],
            "target_resolution": list(self.target_size),
            "channels": 3,
            "mean_brightness": round(mean_brightness, 4),
            "contrast": round(contrast, 4),
            "processed_array": img_np,
            "pil_image": resized
        }

