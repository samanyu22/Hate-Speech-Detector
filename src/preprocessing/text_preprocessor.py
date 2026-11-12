"""
Text Preprocessing Pipeline
Handles text normalization, Unicode cleaning, whitespace handling,
tokenization, and subword/vocabulary encoding.
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional
import torch


class TextPreprocessor:
    """Preprocesses raw text into clean normalized text and token tensor representations."""

    def __init__(self, max_length: int = 128, lowercase: bool = True):
        self.max_length = max_length
        self.lowercase = lowercase
        # Lightweight built-in vocabulary for CPU efficiency
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        self.inv_vocab: Dict[int, str] = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>", 3: "<SEP>"}
        self._build_initial_vocab()

    def _build_initial_vocab(self):
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of",
            "with", "by", "and", "or", "not", "this", "that", "it", "from", "as", "be", "has",
            "have", "had", "speech", "video", "image", "audio", "news", "official", "breaking",
            "report", "claims", "press", "conference", "statement", "president", "minister",
            "announces", "leak", "secret", "fake", "deepfake", "real", "authentic", "footage",
            "footage", "event", "meeting", "summit", "protest", "police", "court", "trial",
            "actor", "actress", "celebrity", "shocking", "viral", "unveiled", "investigation"
        ]
        for w in common_words:
            if w not in self.vocab:
                idx = len(self.vocab)
                self.vocab[w] = idx
                self.inv_vocab[idx] = w

    def normalize(self, text: str) -> str:
        """Applies Unicode normalization, whitespace trimming, and optional lowercasing."""
        if not text:
            return ""
        # Unicode NFKD decomposition
        text = unicodedata.normalize("NFKD", text)
        # Normalize whitespace (replace tabs, newlines, multiple spaces)
        text = re.sub(r"\s+", " ", text).strip()
        if self.lowercase:
            text = text.lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Splits normalized text into tokens while preserving semantic punctuation."""
        clean = self.normalize(text)
        # Separate punctuation into tokens
        tokens = re.findall(r"\w+|[^\w\s]", clean, re.UNICODE)
        return tokens

    def encode(self, text: str) -> Dict[str, Any]:
        """Encodes text into token IDs, attention masks, and provides token details."""
        tokens = self.tokenize(text)
        
        # Build token IDs with CLS and SEP
        token_ids = [self.vocab["<CLS>"]]
        processed_tokens = ["<CLS>"]
        
        for t in tokens[: self.max_length - 2]:
            t_clean = t.lower() if self.lowercase else t
            if t_clean not in self.vocab and len(self.vocab) < 5000:
                idx = len(self.vocab)
                self.vocab[t_clean] = idx
                self.inv_vocab[idx] = t_clean
            token_id = self.vocab.get(t_clean, self.vocab["<UNK>"])
            token_ids.append(token_id)
            processed_tokens.append(t)
            
        token_ids.append(self.vocab["<SEP>"])
        processed_tokens.append("<SEP>")
        
        # Pad to max_length
        seq_len = len(token_ids)
        padding_len = self.max_length - seq_len
        attention_mask = [1] * seq_len + [0] * padding_len
        padded_token_ids = token_ids + [self.vocab["<PAD>"]] * padding_len
        
        return {
            "original_text": text,
            "normalized_text": self.normalize(text),
            "tokens": processed_tokens,
            "token_count": len(tokens),
            "input_ids": torch.tensor(padded_token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

