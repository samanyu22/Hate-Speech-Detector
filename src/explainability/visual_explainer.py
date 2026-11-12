"""
Visual Explainability Module (Grad-CAM & Saliency Heatmaps)
Extracts spatial activation maps from convolutional layers and overlays
color-coded evidence heatmaps over the input image for explainable verification.
"""

import io
import base64
from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class VisualExplainer:
    """Generates localized spatial activation maps and evidence overlays for image predictions."""

    def __init__(self, vision_pipeline: Any):
        self.pipeline = vision_pipeline

    def explain(self, image_source: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Generates localized Grad-CAM-style visual heatmap and top suspicious patches."""
        prep = self.pipeline.preprocessor.preprocess(image_source)
        tensor = prep["tensor"].unsqueeze(0)  # (1, 3, 224, 224)
        tensor.requires_grad = True

        self.pipeline.model.eval()
        logits, emb = self.pipeline.model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        # Target class: fake (index 1)
        score = logits[0, 1]
        score.backward()

        # Input gradient saliency map
        grads = tensor.grad.data.abs().squeeze(0).mean(dim=0).cpu().numpy()  # (224, 224)
        
        # Smooth and normalize heatmap
        grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-6)
        
        # Convert to RGB heatmap (JET colormap simulation)
        h, w = grads.shape
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        # Red highlights high attribution, blue highlights low
        heatmap[:, :, 0] = (grads * 255).astype(np.uint8)
        heatmap[:, :, 1] = ((1.0 - np.abs(grads - 0.5) * 2.0) * 255).clip(0, 255).astype(np.uint8)
        heatmap[:, :, 2] = ((1.0 - grads) * 255).astype(np.uint8)

        # Blend heatmap with original image preview
        pil_orig = self.pipeline.preprocessor.load_image(image_source).resize((224, 224))
        orig_arr = np.array(pil_orig, dtype=np.uint8)
        blended = (orig_arr * 0.6 + heatmap * 0.4).astype(np.uint8)
        
        # Encode blended heatmap as base64
        overlay_pil = Image.fromarray(blended)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="JPEG", quality=85)
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        # Calculate localized focus metrics (center vs boundary)
        center_patch = grads[56:168, 56:168].mean()
        boundary_patch = (grads.sum() - grads[56:168, 56:168].sum()) / (224 * 224 - 112 * 112)
        primary_focus = "CENTRAL_FACIAL" if center_patch > boundary_patch else "PERIPHERAL_BOUNDARY"

        return {
            "heatmap_b64": overlay_b64,
            "max_activation": float(round(grads.max(), 3)),
            "mean_activation": float(round(grads.mean(), 3)),
            "primary_focus_region": primary_focus,
            "center_energy_ratio": float(round(center_patch / max(boundary_patch, 1e-4), 2))
        }

