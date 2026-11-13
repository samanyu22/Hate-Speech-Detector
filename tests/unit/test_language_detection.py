import pytest
from src.language.detector import MultilingualLanguageDetector

def test_english_detection():
    text = "Education is the primary weapon to eliminate poverty in our society."
    info = MultilingualLanguageDetector.detect(text)
    assert info["language"] == "en"
    assert info["is_code_mixed"] is False

def test_devanagari_hindi_detection():
    text = "भारत की विविधता ही हमारी सबसे बड़ी शक्ति और पहचान है।"
    info = MultilingualLanguageDetector.detect(text)
    assert info["language"] == "hi"
    assert info["is_code_mixed"] is False

def test_hinglish_code_mixed_detection():
    text = "Ye saale kaminon ko desh se nikal phenko bhai!"
    info = MultilingualLanguageDetector.detect(text)
    assert info["language"] == "hi-en"
    assert info["is_code_mixed"] is True

