import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Any

class BytePairTokenizer:
    """
    Multilingual Subword Byte-Pair Encoding (BPE) Tokenizer.
    
    Implements Sennrich et al. (ACL 2016) / LLaMA / GPT BPE algorithm:
    - Represents words as sequences of characters + end-of-word marker ('</w>').
    - Iteratively counts adjacent symbol pairs across the training corpus.
    - Merges the most frequent adjacent pair into a new subword unit.
    - Tokenizes unseen / out-of-vocabulary words into recognized subword fragments.
    - Eliminates <UNK> collapse for novel slang, morphologically rich Hindi, and typos.
    """
    def __init__(self, vocab_file: Optional[str] = None):
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.is_trained: bool = False

        if vocab_file and Path(vocab_file).exists():
            self.load(vocab_file)
        else:
            self._init_special_tokens()

    def _init_special_tokens(self):
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.inv_vocab = {idx: tok for idx, tok in enumerate(self.special_tokens)}

    @staticmethod
    def _get_stats(vocab_word_counts: Dict[Tuple[str, ...], int]) -> Counter:
        """Count frequencies of adjacent symbol pairs across all vocabulary words."""
        pairs = Counter()
        for word_tuple, freq in vocab_word_counts.items():
            for i in range(len(word_tuple) - 1):
                pairs[word_tuple[i], word_tuple[i + 1]] += freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: Tuple[str, str], vocab_word_counts: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of the selected pair in all vocabulary words."""
        new_vocab = {}
        bigram = pair
        first, second = bigram

        for word in vocab_word_counts:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = vocab_word_counts[word]
        return new_vocab

    def train(self, texts: List[str], target_vocab_size: int = 5000, min_frequency: int = 2) -> Dict[str, Any]:
        """
        Train BPE merges over a corpus of multilingual texts.
        """
        self._init_special_tokens()

        # Build initial character-level vocabulary with end-of-word marker
        word_freqs = Counter()
        for text in texts:
            # Clean and split into word tokens
            tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            for tok in tokens:
                word_freqs[tok.lower()] += 1

        # Format: tuple of characters + end-of-word delimiter '</w>'
        vocab_word_counts: Dict[Tuple[str, ...], int] = {}
        for word, freq in word_freqs.items():
            if freq >= min_frequency:
                chars = tuple(list(word) + ["</w>"])
                vocab_word_counts[chars] = freq

        # Initialize single characters in vocabulary
        all_chars = set()
        for word_tuple in vocab_word_counts:
            all_chars.update(word_tuple)

        for c in sorted(all_chars):
            if c not in self.vocab:
                idx = len(self.vocab)
                self.vocab[c] = idx
                self.inv_vocab[idx] = c

        self.merges = []
        num_merges = target_vocab_size - len(self.vocab)

        for step in range(max(0, num_merges)):
            pairs = self._get_stats(vocab_word_counts)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            vocab_word_counts = self._merge_vocab(best_pair, vocab_word_counts)
            self.merges.append(best_pair)
            
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx
                self.inv_vocab[idx] = merged_token

        self.is_trained = True
        return {
            "status": "trained",
            "vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "original_unique_words": len(word_freqs)
        }

    def tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subword tokens using learned BPE merges.
        Unseen words are decomposed into subword pieces without collapsing to <UNK>.
        """
        if not word:
            return []

        # Start with individual characters + end-of-word marker
        word_tokens = list(word.lower()) + ["</w>"]

        for pair in self.merges:
            first, second = pair
            i = 0
            new_tokens = []
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == first and word_tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_tokens
            if len(word_tokens) == 1:
                break

        return word_tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize full text into a list of BPE subword tokens.
        """
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        subwords = []
        for w in words:
            subwords.extend(self.tokenize_word(w))
        return subwords

    def encode(self, text: str) -> List[int]:
        """
        Encode raw text into integer subword token IDs.
        """
        subwords = self.tokenize(text)
        unk_id = self.vocab.get("<UNK>", 1)
        return [self.vocab.get(sw, unk_id) for sw in subwords]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode integer subword token IDs back into natural text.
        """
        tokens = [self.inv_vocab.get(i, "<UNK>") for i in token_ids]
        raw_text = "".join(tokens).replace("</w>", " ").strip()
        return raw_text

    def save(self, filepath: str):
        """Save BPE vocabulary and learned merge rules to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Load BPE vocabulary and learned merge rules from disk."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = [tuple(m) for m in data["merges"]]
        self.special_tokens = data.get("special_tokens", self.special_tokens)
        self.is_trained = True
