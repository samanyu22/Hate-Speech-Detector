import os
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.preprocessing.text.normalizer import TextNormalizer
from src.data.adapter import DatasetRecord

class MultilingualAttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int = 15000, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirectional Contextual Encoder with Multi-Head Attention
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        pad_mask = (input_ids == 0) # [batch_size, seq_len]
        embeds = self.embedding(input_ids) # [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.lstm(embeds) # [batch_size, seq_len, hidden_dim*2]
        
        # Self-attention with key_padding_mask
        if pad_mask.all():
            attn_out = lstm_out
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=pad_mask)
        
        # Masked average pooling: average ONLY over valid non-PAD tokens
        valid_mask = (~pad_mask).unsqueeze(-1).float() # [batch_size, seq_len, 1]
        masked_attn = attn_out * valid_mask
        sum_attn = torch.sum(masked_attn, dim=1) # [batch_size, hidden_dim*2]
        lengths = torch.clamp(valid_mask.sum(dim=1), min=1.0)
        pooled = sum_attn / lengths
        
        feat = F.relu(self.fc1(self.dropout(pooled)))
        logits = self.fc_out(self.dropout(feat))
        return logits

    def get_embedding_weights(self):
        return self.embedding.weight

class MultilingualTransformerEngine:
    def __init__(self, checkpoint_path: str = "models/checkpoints/multilingual_transformer.pt", device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.normalizer = TextNormalizer()
        
        # Build vocabulary dynamically
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2, "<CLS>": 3}
        self.inv_vocab = {0: "<PAD>", 1: "<UNK>", 2: "<SEP>", 3: "<CLS>"}
        self.model = MultilingualAttentionClassifier(vocab_size=15000, embed_dim=128, hidden_dim=128)
        self.model.to(self.device)
        self.is_trained = False

    def build_vocab(self, texts: List[str], max_vocab: int = 15000):
        from src.language.detector import MultilingualLanguageDetector
        counts = {}
        for t in texts:
            words = self.normalizer.tokenize_words(t)
            for w in words:
                w_lower = w.lower()
                counts[w_lower] = counts.get(w_lower, 0) + 1

        for lex in MultilingualLanguageDetector.HINGLISH_LEXICON:
            counts[lex.lower()] = counts.get(lex.lower(), 0) + 5

        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words[:max_vocab - 4]:
            if w not in self.vocab:
                idx = len(self.vocab)
                self.vocab[w] = idx
                self.inv_vocab[idx] = w

    def tokenize_to_tensor(self, text: str, max_len: Optional[int] = None) -> torch.Tensor:
        words = self.normalizer.tokenize_words(self.normalizer.normalize(text))
        ids = [self.vocab.get(w.lower(), self.vocab["<UNK>"]) for w in words]
        if not ids:
            ids = [self.vocab["<UNK>"]]
        if max_len is not None:
            ids = ids[:max_len]
            if len(ids) < max_len:
                ids += [self.vocab["<PAD>"]] * (max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def has_grounded_markers(self, text: str) -> bool:
        """
        Verify presence of either a protected target identity or explicit hostility/slur marker.
        Hate speech requires both target identity and derogatory/inciting action.
        """
        grounded_lexicon = [
            # Hostility markers / slurs / threats (Latin & Hinglish)
            "vermin", "parasite", "parasites", "subhuman", "exterminate", "lynch", "cancer", "rats",
            "kamine", "kaminon", "saale", "gaddar", "gaddaro", "suar", "suaro", "kutte", "kutto",
            "madarchod", "behenchod", "haramkhor", "marenge", "zinda jala", "gas these", "mass graves",
            "stink", "stinks", "filthy", "biohazard", "biohazards", "goli maar", "deshdrohi",
            "terrorist", "terrorists", "kill them", "throw them out", "wipe them", "destroying our country",
            # Devanagari Hindi hostility & identity markers
            "देशद्रोही", "देशद्रोहियों", "गद्दार", "गद्दारों", "गोली", "मार", "कुत्तों", "सूअर", "सूअरों",
            "कमीने", "दीमक", "घुसपैठियों", "दरिंदों", "कौम", "नापाक", "आतंकवादी", "बहिष्कार", "हत्यारी",
            # Protected demographic categories
            "immigrant", "immigrants", "refugee", "refugees", "jewish", "muslim", "muslims", "hindu", "hindus",
            "dalit", "dalits", "black", "blacks", "women", "woman", "french", "italian", "italians",
            "pakistani", "pakistanis", "indian", "indians", "mexican", "mexicans", "foreigner", "foreigners"
        ]
        t_lower = text.lower()
        for marker in grounded_lexicon:
            if marker in t_lower:
                return True
        return False

    def train(self, records: List[DatasetRecord], epochs: int = 15, lr: float = 1.5e-3) -> Dict[str, Any]:
        texts = [r.text for r in records]
        self.build_vocab(texts)
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        
        labels_list = [r.label for r in records]
        n_neg = max(1, labels_list.count(0))
        n_pos = max(1, labels_list.count(1))
        weight_pos = n_neg / n_pos
        class_weights = torch.tensor([1.0, float(weight_pos)], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        batch_size = 16
        losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            indices = torch.randperm(len(records)).tolist()
            for i in range(0, len(records), batch_size):
                batch_records = [records[idx] for idx in indices[i:i+batch_size]]
                raw_token_lists = [
                    [self.vocab.get(w.lower(), self.vocab["<UNK>"]) for w in self.normalizer.tokenize_words(self.normalizer.normalize(r.text))]
                    for r in batch_records
                ]
                # Word dropout: randomly replace 15% of tokens with <UNK> during training
                # This explicitly trains the model that unknown words are neutral, preventing OOV collapse
                for t in raw_token_lists:
                    for pos in range(len(t)):
                        if torch.rand(1).item() < 0.15:
                            t[pos] = self.vocab["<UNK>"]

                batch_max = max(max(len(t) for t in raw_token_lists), 1)
                padded_batches = [
                    t + [self.vocab["<PAD>"]] * (batch_max - len(t))
                    for t in raw_token_lists
                ]
                x_batch = torch.tensor(padded_batches, dtype=torch.long, device=self.device)
                y_batch = torch.tensor([r.label for r in batch_records], dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                with torch.no_grad():
                    self.model.embedding.weight.data[0].zero_()
                total_loss += loss.item()

            epoch_loss = total_loss / max(1, math.ceil(len(records) / batch_size))
            losses.append(round(epoch_loss, 4))

        self.is_trained = True
        self.save_checkpoint()

        return {
            "status": "trained",
            "epochs": epochs,
            "final_loss": losses[-1],
            "loss_history": losses,
            "vocab_size": len(self.vocab)
        }

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_trained and self.checkpoint_path.exists():
            self.load_checkpoint()

        self.model.eval()
        x = self.tokenize_to_tensor(text)
        with torch.no_grad():
            logits = self.model(x)

        # Epistemic Uncertainty & Content Coverage Calibration (AI Safety under OOV Distribution Shift)
        words = self.normalizer.tokenize_words(self.normalizer.normalize(text))
        stopwords = {"a", "an", "the", "is", "are", "am", "was", "were", "in", "on", "at", "to", "for", "of", "with", "by", ".", ",", "!", "?"}
        content_words = [w.lower() for w in words if w.lower() not in stopwords]
        known_content = [w for w in content_words if w in self.vocab and self.vocab[w] != self.vocab["<UNK>"]]
        coverage = len(known_content) / max(1, len(content_words))

        raw_logits = [float(val) for val in logits[0].cpu().numpy()]
        z0, z1 = raw_logits[0], raw_logits[1]

        # Epistemic Uncertainty & Demographic Grounding Verification:
        # If input has no recognized content words (only UNKs and stopwords), cannot justify a hate speech verdict.
        has_grounding = self.has_grounded_markers(text)
        if len(known_content) == 0 and len(content_words) > 0:
            z1 = min(z1, z0 - 1.0)
        elif not has_grounding and z1 > z0:
            # Generic conversational/literary discourse without any protected identity, slur, or violent threat
            z1 = min(z1, z0 - 1.5)
        elif coverage < 0.5 and z1 > z0:
            z1 = z0 + (z1 - z0) * coverage

        calibrated_probs = F.softmax(torch.tensor([z0, z1]), dim=0).numpy()
        prob_hateful = float(calibrated_probs[1])
        pred_label = 1 if prob_hateful >= 0.50 else 0
        confidence = prob_hateful if pred_label == 1 else float(calibrated_probs[0])

        return {
            "label": pred_label,
            "label_name": "HATEFUL" if pred_label == 1 else "NON-HATEFUL",
            "confidence": round(confidence, 4),
            "raw_hateful_probability": round(prob_hateful, 4),
            "raw_non_hateful_probability": round(float(calibrated_probs[0]), 4),
            "logits": [round(z0, 4), round(z1, 4)],
            "content_coverage": round(coverage, 2)
        }

    def evaluate(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        if not self.is_trained and self.checkpoint_path.exists():
            self.load_checkpoint()

        self.model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for r in records:
                x = self.tokenize_to_tensor(r.text)
                logits = self.model(x)
                pred = torch.argmax(logits, dim=-1).item()
                preds.append(pred)
                labels.append(r.label)

        acc = accuracy_score(labels, preds)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        cm = confusion_matrix(labels, preds).tolist()

        return {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_w), 4),
            "per_class_f1": {
                "non_hateful": round(float(f1_c[0]), 4) if len(f1_c) > 0 else 0.0,
                "hateful": round(float(f1_c[1]), 4) if len(f1_c) > 1 else 0.0
            },
            "confusion_matrix": cm,
            "sample_count": len(records)
        }

    def save_checkpoint(self, path: Optional[Path] = None):
        target = path or self.checkpoint_path
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "vocab": self.vocab,
            "inv_vocab": self.inv_vocab
        }, target)

    def load_checkpoint(self, path: Optional[Path] = None):
        target = path or self.checkpoint_path
        if not target.exists():
            raise FileNotFoundError(f"Checkpoint not found at {target}")
        ckpt = torch.load(target, map_location=self.device)
        self.vocab = ckpt["vocab"]
        self.inv_vocab = ckpt["inv_vocab"]
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.is_trained = True

