import pytest
from src.preprocessing.text.normalizer import TextNormalizer

def test_unicode_and_url_normalization():
    norm = TextNormalizer()
    raw = "Check this out https://spam.com/link and visit @JohnDoe"
    cleaned = norm.normalize(raw)
    assert "[URL]" in cleaned
    assert "[MENTION]" in cleaned
    assert "https://" not in cleaned

def test_hashtag_segmentation():
    norm = TextNormalizer()
    raw = "We stand for #StopHate and #PeaceForAll"
    cleaned = norm.normalize(raw)
    assert "Stop Hate" in cleaned
    assert "Peace For All" in cleaned

def test_emoji_and_whitespace():
    norm = TextNormalizer()
    raw = "Great   match   dost!   "
    cleaned = norm.normalize(raw)
    assert cleaned == "Great match dost!"

