"""
Feature Extraction and Modality Encoders
Implements neural encoders for Text, Vision, and Audio to generate
projected 128-dimensional embedding representations.
"""

import math
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Encodes token sequences into a 128-dimensional semantic text representation."""
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: (B, L)
        emb = self.embedding(input_ids)  # (B, L, E)
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1)
            
        emb_trans = emb.transpose(1, 2)  # (B, E, L)
        c1 = F.relu(self.conv1(emb_trans))  # (B, H, L)
        c2 = F.relu(self.conv2(emb_trans))  # (B, H, L)
        
        # Max-pooling across sequence length
        p1 = F.adaptive_max_pool1d(c1, 1).squeeze(-1)  # (B, H)
        p2 = F.adaptive_max_pool1d(c2, 1).squeeze(-1)  # (B, H)
        
        cat = torch.cat([p1, p2], dim=-1)  # (B, 2H)
        out = self.norm(F.relu(self.fc(cat)))  # (B, hidden_dim)
        return out


class VisionEncoder(nn.Module):
    """Encodes 224x224 images into a 128-dimensional visual feature representation."""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )
        self.fc = nn.Linear(128, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        feat = self.features(x)
        flat = torch.flatten(feat, 1)
        out = self.norm(F.relu(self.fc(flat)))
        return out


class AudioEncoder(nn.Module):
    """Encodes (64, 96) Mel-spectrograms into a 128-dimensional acoustic representation."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 48)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 24)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # (128, 1, 1)
        )
        self.fc = nn.Linear(128, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        feat = self.features(x)
        flat = torch.flatten(feat, 1)
        out = self.norm(F.relu(self.fc(flat)))
        return out
