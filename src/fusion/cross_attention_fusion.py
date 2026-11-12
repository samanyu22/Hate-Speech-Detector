"""
Cross-Attention Multimodal Fusion Architecture
Implements multi-head cross-attention across text, visual, and audio representations
to learn mutual dependencies and cross-modal attention maps.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusionModel(nn.Module):
    """Bidirectional multi-head cross-attention fusion network."""

    def __init__(self, emb_dim: int = 128, num_heads: int = 4, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Modality projection
        self.proj_t = nn.Linear(emb_dim, emb_dim)
        self.proj_v = nn.Linear(emb_dim, emb_dim)
        self.proj_a = nn.Linear(emb_dim, emb_dim)
        
        # Learned missing modality tokens
        self.empty_t = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.empty_v = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.empty_a = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        
        # Multi-head attention across modality sequence (sequence length 3: [T, V, A])
        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        
        # Feedforward classification head
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        t_emb: Optional[torch.Tensor] = None,
        v_emb: Optional[torch.Tensor] = None,
        a_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for emb in [t_emb, v_emb, a_emb]:
            if emb is not None:
                bs = emb.shape[0]
                break
        else:
            raise ValueError("At least one modality embedding must be provided.")

        t = self.proj_t(t_emb).unsqueeze(1) if t_emb is not None else self.empty_t.expand(bs, 1, -1)
        v = self.proj_v(v_emb).unsqueeze(1) if v_emb is not None else self.empty_v.expand(bs, 1, -1)
        a = self.proj_a(a_emb).unsqueeze(1) if a_emb is not None else self.empty_a.expand(bs, 1, -1)

        # Sequence of tokens: (B, 3, emb_dim)
        seq = torch.cat([t, v, a], dim=1)
        
        # Self/Cross attention across modalities
        attn_out, attn_weights = self.mha(seq, seq, seq)
        res = self.norm(seq + attn_out)  # (B, 3, emb_dim)
        
        flat = res.reshape(bs, -1)  # (B, 3 * emb_dim)
        logits = self.classifier(flat)
        
        return logits, attn_weights

