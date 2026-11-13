import re
from typing import Dict, Any, Tuple

class MultilingualLanguageDetector:
    # Common Romanized Hindi lexicon for code-mixing detection
    HINGLISH_LEXICON = {
        "hai", "hain", "nahi", "nahin", "kya", "kyu", "kyun", "karo", "karna", "raha", "rahe", "rahi",
        "bhai", "desh", "deshdrohi", "gaddar", "gaddaro", "kamine", "kaminon", "saale", "saala", "sabko",
        "tera", "meri", "hum", "tum", "aap", "aur", "bhi", "toh", "bahut", "aukaat", "aukat", "suar",
        "suaro", "kutte", "kutto", "janwar", "chor", "dhanyawad", "shubhkaamnayein", "pyaar", "dost",
        "madarchod", "behenchod", "haramkhor", "basti", "marenge", "jala", "nikalo", "phenko", "bojh",
        "aaj", "kal", "dekh", "mast", "sabse", "zindagi", "samay", "acche", "goli", "bhookh"
    }

    @classmethod
    def detect(cls, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_code_mixed": False,
                "script_breakdown": {"devanagari": 0.0, "latin": 0.0, "other": 0.0}
            }

        # Analyze characters
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        total_alpha = devanagari_chars + latin_chars

        if total_alpha == 0:
            return {
                "language": "en",
                "confidence": 0.50,
                "is_code_mixed": False,
                "script_breakdown": {"devanagari": 0.0, "latin": 0.0, "other": 1.0}
            }

        dev_ratio = devanagari_chars / total_alpha
        latin_ratio = latin_chars / total_alpha

        # Analyze tokens for Romanized Hindi (Hinglish)
        tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
        hinglish_hits = sum(1 for t in tokens if t in cls.HINGLISH_LEXICON)
        hinglish_ratio = (hinglish_hits / len(tokens)) if tokens else 0.0

        # Classification rules
        if dev_ratio > 0.60:
            if latin_ratio > 0.15:
                lang = "hi-en"
                is_cm = True
                conf = 0.85
            else:
                lang = "hi"
                is_cm = False
                conf = 0.95
        elif latin_ratio > 0.60:
            if hinglish_ratio >= 0.15 or dev_ratio > 0.05:
                lang = "hi-en"
                is_cm = True
                conf = min(0.95, 0.60 + hinglish_ratio)
            else:
                lang = "en"
                is_cm = False
                conf = 0.90
        else:
            lang = "hi-en"
            is_cm = True
            conf = 0.80

        return {
            "language": lang,
            "confidence": round(conf, 3),
            "is_code_mixed": is_cm,
            "script_breakdown": {
                "devanagari": round(dev_ratio, 3),
                "latin": round(latin_ratio, 3),
                "hinglish_lexicon_ratio": round(hinglish_ratio, 3)
            }
        }

