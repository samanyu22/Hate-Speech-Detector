import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class VisualFeatureExtractor(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        # Lightweight convolutional visual feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 3, 224, 224]
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)

class FeatureConcatFusion(nn.Module):
    def __init__(self, text_dim: int = 128, visual_dim: int = 128, context_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc_fused = nn.Sequential(
            nn.Linear(text_dim + visual_dim + context_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, t_feat: torch.Tensor, v_feat: torch.Tensor, c_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([t_feat, v_feat, c_feat], dim=-1)
        return self.fc_fused(fused)

class DecisionFusion(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Learnable modality weighting
        self.weights = nn.Parameter(torch.tensor([0.50, 0.35, 0.15])) # text, visual, context

    def forward(self, t_logits: torch.Tensor, v_logits: torch.Tensor, c_logits: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.weights, dim=0)
        return w[0] * t_logits + w[1] * v_logits + w[2] * c_logits

class CrossModalCrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 128, num_heads: int = 4, num_classes: int = 2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Linear(dim * 2, dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, t_feat: torch.Tensor, v_feat: torch.Tensor) -> torch.Tensor:
        # t_feat: [batch, 1, dim], v_feat: [batch, 1, dim]
        attn_out, _ = self.cross_attn(query=t_feat, key=v_feat, value=v_feat)
        g = torch.sigmoid(self.gate(torch.cat([t_feat, attn_out], dim=-1)))
        fused = g * t_feat + (1 - g) * attn_out
        logits = self.classifier(fused.squeeze(1))
        return logits

class MultimodalDetector(nn.Module):
    def __init__(self, dim: int = 128, fusion_type: str = "cross_attention"):
        super().__init__()
        self.dim = dim
        self.fusion_type = fusion_type
        self.visual_extractor = VisualFeatureExtractor(out_dim=dim)
        self.context_fc = nn.Linear(dim, dim)
        self.cross_attention = CrossModalCrossAttentionFusion(dim=dim)
        self.concat_fusion = FeatureConcatFusion(text_dim=dim, visual_dim=dim, context_dim=dim)
        self.fallback_classifier = nn.Linear(dim, 2)

    def forward(
        self,
        text_feat: torch.Tensor,
        image_tensor: Optional[torch.Tensor] = None,
        context_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # If image is missing, gracefully fall back to text representation
        if image_tensor is None:
            return self.fallback_classifier(text_feat)

        v_feat = self.visual_extractor(image_tensor)

        if self.fusion_type == "cross_attention":
            t_seq = text_feat.unsqueeze(1)
            v_seq = v_feat.unsqueeze(1)
            return self.cross_attention(t_seq, v_seq)
        elif self.fusion_type == "feature_concat":
            c_feat = self.context_fc(context_feat) if context_feat is not None else torch.zeros_like(text_feat)
            return self.concat_fusion(text_feat, v_feat, c_feat)
        else:
            # Fallback
            return self.fallback_classifier(text_feat)

