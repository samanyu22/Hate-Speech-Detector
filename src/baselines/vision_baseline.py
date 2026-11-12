"""
Vision-Only Deepfake / Manipulated Imagery Baseline Model
Provides an independent vision-only classification pipeline with real training,
validation, checkpoint management, and inference.
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.features.feature_extractor import VisionEncoder


class VisionClassifier(nn.Module):
    """Neural network classifier using VisionEncoder + MLP classification head."""
    def __init__(self, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.encoder = VisionEncoder(hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, emb


class VisionBaselinePipeline:
    """Manages training, evaluation, and inference for the vision baseline."""

    def __init__(self, checkpoint_path: str = "models/checkpoints/vision_baseline.pt"):
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.preprocessor = ImagePreprocessor()
        self.model = VisionClassifier()
        self.is_trained = False
        
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.is_trained = True
        except Exception as e:
            print(f"Notice: Could not load vision checkpoint: {e}")

    def save_checkpoint(self, path: Optional[str] = None):
        target = path or self.checkpoint_path
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": "vision_baseline",
            "is_trained": True
        }, target)
        self.is_trained = True

    def fit(self, train_samples: List[Any], val_samples: List[Any], epochs: int = 5, lr: float = 0.001) -> Dict[str, Any]:
        """Trains the vision classifier using actual image files from dataset records."""
        def prepare_data(samples):
            data = []
            for s in samples:
                if s.image_path and os.path.exists(s.image_path):
                    try:
                        prep = self.preprocessor.preprocess(s.image_path)
                        label_idx = 0 if s.label == "REAL" else 1
                        data.append({
                            "tensor": prep["tensor"],
                            "label": torch.tensor(label_idx, dtype=torch.long)
                        })
                    except Exception:
                        continue
            return data

        train_data = prepare_data(train_samples)
        val_data = prepare_data(val_samples)
        
        if not train_data:
            raise ValueError("No valid image data available for vision training.")
            
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        history = {"loss": [], "val_acc": []}
        
        for ep in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                logits, _ = self.model(batch["tensor"])
                loss = criterion(logits, batch["label"])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Val accuracy
            self.model.eval()
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    logits, _ = self.model(batch["tensor"])
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == batch["label"]).sum().item()
            val_acc = correct / max(len(val_data), 1)
            history["loss"].append(round(total_loss / max(len(train_loader), 1), 4))
            history["val_acc"].append(round(val_acc, 4))
            
        self.save_checkpoint()
        return history

    def predict(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Predicts REAL or FAKE for raw image input."""
        self.model.eval()
        prep = self.preprocessor.preprocess(image_input)
        img_tensor = prep["tensor"].unsqueeze(0)  # (1, 3, 224, 224)
        
        with torch.no_grad():
            logits, emb = self.model(img_tensor)
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
            "preview_b64": prep["preview_b64"],
            "resolution": prep["original_resolution"]
        }

