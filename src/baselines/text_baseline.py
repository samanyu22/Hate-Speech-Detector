"""
Text-Only Deepfake / Misinformation Baseline Model
Provides an independent text-only classification pipeline with real training,
validation, checkpoint management, and inference.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.features.feature_extractor import TextEncoder


class TextClassifier(nn.Module):
    """Neural network classifier using TextEncoder + MLP classification head."""
    def __init__(self, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.encoder = TextEncoder(hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(input_ids, attention_mask)
        logits = self.classifier(emb)
        return logits, emb


class TextBaselinePipeline:
    """Manages training, evaluation, and inference for the text baseline."""

    def __init__(self, checkpoint_path: str = "models/checkpoints/text_baseline.pt"):
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.preprocessor = TextPreprocessor()
        self.model = TextClassifier()
        self.is_trained = False
        
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.is_trained = True
        except Exception as e:
            print(f"Notice: Could not load text checkpoint: {e}")

    def save_checkpoint(self, path: Optional[str] = None):
        target = path or self.checkpoint_path
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": "text_baseline",
            "is_trained": True
        }, target)
        self.is_trained = True

    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            logits, _ = self.model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)

    def fit(self, train_samples: List[Any], val_samples: List[Any], epochs: int = 5, lr: float = 0.001) -> Dict[str, Any]:
        """Trains the text classifier using actual dataset records."""
        # Build dataset
        def prepare_data(samples):
            data = []
            for s in samples:
                enc = self.preprocessor.encode(s.text)
                label_idx = 0 if s.label == "REAL" else 1
                data.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "label": torch.tensor(label_idx, dtype=torch.long)
                })
            return data

        train_data = prepare_data(train_samples)
        val_data = prepare_data(val_samples)
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {"loss": [], "val_acc": []}
        
        for ep in range(epochs):
            loss = self.train_epoch(train_loader, optimizer, criterion)
            # Evaluate val accuracy
            self.model.eval()
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    logits, _ = self.model(batch["input_ids"], batch["attention_mask"])
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == batch["label"]).sum().item()
            val_acc = correct / max(len(val_data), 1)
            history["loss"].append(round(loss, 4))
            history["val_acc"].append(round(val_acc, 4))
            
        self.save_checkpoint()
        return history

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts REAL or FAKE for raw input text."""
        self.model.eval()
        enc = self.preprocessor.encode(text)
        with torch.no_grad():
            logits, emb = self.model(enc["input_ids"].unsqueeze(0), enc["attention_mask"].unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            
        p_real = float(probs[0].item())
        p_fake = float(probs[1].item())
        pred_label = "FAKE" if p_fake > p_real else "REAL"
        confidence = max(p_real, p_fake)
        
        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "REAL": round(p_real, 4),
                "FAKE": round(p_fake, 4)
            },
            "embedding": emb.squeeze(0),
            "tokens": enc["tokens"]
        }

