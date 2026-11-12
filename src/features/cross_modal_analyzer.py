"""
Cross-Modal Consistency Analyzer
Learns and evaluates cross-modal alignment, contradictions, and semantic mismatches
between text, visual, and audio modalities via contrastive projection heads.
"""

import os
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CrossModalAnalyzer(nn.Module):
    """Evaluates cross-modal alignment, contradictions, and semantic mismatches."""

    def __init__(self, emb_dim: int = 128, proj_dim: int = 64, mismatch_threshold: float = 0.05):
        super().__init__()
        self.mismatch_threshold = mismatch_threshold
        self.proj_dim = proj_dim
        
        # Projection layers into joint cross-modal semantic space
        self.align_t = nn.Linear(emb_dim, proj_dim)
        self.align_v = nn.Linear(emb_dim, proj_dim)
        self.align_a = nn.Linear(emb_dim, proj_dim)
        self.is_calibrated = False

    def forward_t(self, t_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.align_t(t_emb), p=2, dim=-1)

    def forward_v(self, v_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.align_v(v_emb), p=2, dim=-1)

    def forward_a(self, a_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.align_a(a_emb), p=2, dim=-1)

    def fit_alignment(
        self,
        samples: List[Any],
        text_pipeline: Any,
        vision_pipeline: Any,
        audio_pipeline: Any,
        epochs: int = 4,
        lr: float = 0.005
    ):
        """Calibrates cross-modal alignment heads using contrastive cosine loss on dataset."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for ep in range(epochs):
            for s in samples:
                if not s.text or not s.image_path or not os.path.exists(s.image_path):
                    continue
                try:
                    with torch.no_grad():
                        t_emb = text_pipeline.predict(s.text)["embedding"].unsqueeze(0)
                        v_emb = vision_pipeline.predict(s.image_path)["embedding"].unsqueeze(0)
                    
                    z_t = self.forward_t(t_emb)
                    z_v = self.forward_v(v_emb)
                    sim = torch.sum(z_t * z_v, dim=-1)

                    # Target: 1.0 for genuine, 0.0 for fake/mismatched
                    target_sim = 1.0 if s.label == "REAL" and s.manipulation_type == "authentic" else -0.2
                    loss = F.mse_loss(sim, torch.tensor([target_sim], dtype=torch.float32))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except Exception:
                    continue

        self.is_calibrated = True

    def analyze(
        self,
        text_emb: Optional[torch.Tensor],
        vision_emb: Optional[torch.Tensor],
        audio_emb: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Calculates cross-modal similarity metrics and contradiction diagnostics."""
        results = {
            "text_vision_similarity": None,
            "text_audio_similarity": None,
            "vision_audio_similarity": None,
            "cross_modal_contradiction_detected": False,
            "contradiction_severity": "NONE",
            "findings": []
        }

        # 1. Text vs Vision alignment
        if text_emb is not None and vision_emb is not None:
            z_t = self.forward_t(text_emb.view(1, -1))
            z_v = self.forward_v(vision_emb.view(1, -1))
            tv_sim = float(torch.sum(z_t * z_v).item())
            results["text_vision_similarity"] = round(tv_sim, 4)
            
            if tv_sim < self.mismatch_threshold:
                results["cross_modal_contradiction_detected"] = True
                results["contradiction_severity"] = "HIGH" if tv_sim < 0.0 else "MEDIUM"
                results["findings"].append(
                    f"Semantic mismatch detected between text and visual content (Similarity: {round(tv_sim, 3)} < {self.mismatch_threshold}). The text describes entities/events not supported by visual evidence."
                )
            else:
                results["findings"].append(
                    f"Text and visual modalities exhibit semantic concordance (Similarity: {round(tv_sim, 3)})."
                )

        # 2. Text vs Audio alignment
        if text_emb is not None and audio_emb is not None:
            z_t = self.forward_t(text_emb.view(1, -1))
            z_a = self.forward_a(audio_emb.view(1, -1))
            ta_sim = float(torch.sum(z_t * z_a).item())
            results["text_audio_similarity"] = round(ta_sim, 4)
            
            if ta_sim < self.mismatch_threshold:
                results["cross_modal_contradiction_detected"] = True
                results["findings"].append(
                    f"Acoustic-linguistic divergence detected (Similarity: {round(ta_sim, 3)}). The acoustic characteristics deviate from expected linguistic prosody."
                )

        # 3. Vision vs Audio alignment
        if vision_emb is not None and audio_emb is not None:
            z_v = self.forward_v(vision_emb.view(1, -1))
            z_a = self.forward_a(audio_emb.view(1, -1))
            va_sim = float(torch.sum(z_v * z_a).item())
            results["vision_audio_similarity"] = round(va_sim, 4)

        if not results["findings"]:
            results["findings"].append("Single modality provided; cross-modal consistency analysis requires at least two modalities.")

        return results
