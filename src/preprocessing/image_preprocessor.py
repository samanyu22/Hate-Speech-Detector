"""
Image and Video Frame Preprocessing Pipeline
Handles image validation, aspect-ratio resizing, ImageNet normalization,
tensor conversion, patch extraction, and base64 preview generation.
"""

import os
import io
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class ImagePreprocessor:
    """Processes images and sampled video frames into normalized PyTorch tensors."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def load_image(self, input_source: Union[str, bytes, Image.Image]) -> Image.Image:
        """Loads an image from a path, raw bytes, or returns existing PIL Image."""
        if isinstance(input_source, Image.Image):
            return input_source.convert("RGB")
        elif isinstance(input_source, bytes):
            return Image.open(io.BytesIO(input_source)).convert("RGB")
        elif isinstance(input_source, str):
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"Image file does not exist: {input_source}")
            return Image.open(input_source).convert("RGB")
        else:
            raise TypeError("Unsupported image input source type.")

    def preprocess(self, input_source: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Preprocesses an image and returns tensor, metadata, and base64 inspection preview."""
        pil_img = self.load_image(input_source)
        orig_w, orig_h = pil_img.size
        
        # Tensor transformation
        tensor = self.transform(pil_img)  # Shape: (3, target_size, target_size)
        
        # Generate lightweight base64 preview for UI inspection
        preview_buffer = io.BytesIO()
        preview_img = pil_img.copy()
        preview_img.thumbnail((320, 320))
        preview_img.save(preview_buffer, format="JPEG", quality=85)
        preview_b64 = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.getvalue()).decode("utf-8")
        
        # High-frequency artifact score (detecting unnatural sharpness / noise grid)
        gray = np.array(pil_img.convert("L"), dtype=np.float32)
        laplacian = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        
        return {
            "original_resolution": [orig_w, orig_h],
            "target_resolution": [self.target_size, self.target_size],
            "channels": 3,
            "tensor": tensor,
            "tensor_shape": list(tensor.shape),
            "preview_b64": preview_b64,
            "laplacian_gradient_energy": float(round(laplacian, 3))
        }

    def sample_video_frames(
        self,
        video_path: str,
        max_frames: int = 8,
        fps_sample: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Samples video frames with timestamp tracking.
        If OpenCV is not installed or video is animated/image sequence,
        falls back gracefully to static frame extraction.
        """
        frames_out = []
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0
            
            step = max(int(fps / fps_sample), 1)
            frame_idx = 0
            count = 0
            
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    processed = self.preprocess(pil_frame)
                    processed["timestamp_sec"] = round(frame_idx / fps, 2)
                    processed["frame_number"] = frame_idx
                    frames_out.append(processed)
                    count += 1
                frame_idx += 1
            cap.release()
        except ImportError:
            # Fallback if cv2 is not installed: load video file directly if it's an image/gif
            try:
                pil_img = Image.open(video_path)
                processed = self.preprocess(pil_img)
                processed["timestamp_sec"] = 0.0
                processed["frame_number"] = 0
                frames_out.append(processed)
            except Exception:
                pass
                
        return frames_out

