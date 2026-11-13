import re
import unicodedata
from typing import Dict, Any

class TextNormalizer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.unicode_form = self.config.get("unicode_normalization", "NFKC")
        self.clean_urls = self.config.get("clean_urls", True)
        self.preserve_emojis = self.config.get("preserve_emojis", True)

        # Regex patterns
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@[\w_]+")
        self.hashtag_pattern = re.compile(r"#([\w_]+)")
        self.multiple_spaces = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        # 1. Unicode normalization
        text = unicodedata.normalize(self.unicode_form, text)

        # 2. URL handling
        if self.clean_urls:
            text = self.url_pattern.sub(" [URL] ", text)

        # 3. Mention handling (preserve structure as token)
        text = self.mention_pattern.sub(" [MENTION] ", text)

        # 4. Hashtag handling (split camel case or preserve tag)
        def _split_hashtag(match):
            tag = match.group(1)
            # Split camelCase e.g. StopHate -> Stop Hate
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", tag)
            return f" {spaced} "

        text = self.hashtag_pattern.sub(_split_hashtag, text)

        # 5. Collapse excessive whitespace while preserving line rhythm
        text = self.multiple_spaces.sub(" ", text).strip()

        return text

    def tokenize_words(self, text: str):
        # Basic multilingual word tokenizer
        return re.findall(r"[\w']+|[^\s\w]", text, re.UNICODE)

