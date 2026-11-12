"""
Feature-Level Multimodal Fusion Architecture
Combines Text, Visual, and Audio embeddings via projection, bilinear interaction,
and a deep MLP classification head with missing-modality masking.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureLevelFusionModel(nn.Module):
    """Early/Mid feature fusion with bilinear cross-modal interaction terms."""

    def __init__(self, emb_dim: int = 128, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Modality projection layers
        self.proj_t = nn.Linear(emb_dim, emb_dim)
        self.proj_v = nn.Linear(emb_dim, emb_dim)
        self.proj_a = nn.Linear(emb_dim, emb_dim)
        
        # Learned missing-modality replacement embeddings
        self.empty_v = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.empty_a = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.empty_t = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        
        # Feature combination: [t, v, a, t*v, t*a] -> 5 * 128 = 640
        in_features = emb_dim * 5
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(
        self,
        t_emb: Optional[torch.Tensor] = None,
        v_emb: Optional[torch.Tensor] = None,
        a_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Determine batch size
        for emb in [t_emb, v_emb, a_emb]:
            if emb is not None:
                bs = emb.shape[0]
                break
        else:
            raise ValueError("At least one modality embedding must be provided.")

        # Substitute missing modalities with learned parameters
        t = F.relu(self.proj_t(t_emb)) if t_emb is not None else self.empty_t.expand(bs, -1)
        v = F.relu(self.proj_v(v_emb)) if v_emb is not None else self.empty_v.expand(bs, -1)
        a = F.relu(self.proj_a(a_emb)) if a_emb is not None else self.empty_a.expand(bs, -1)

        # Cross-bilinear interaction terms
        tv_prod = t * v
        ta_prod = t * a

        fused = torch.cat([t, v, a, tv_prod, ta_prod], dim=-1)
        logits = self.mlp(fused)
        return logits

