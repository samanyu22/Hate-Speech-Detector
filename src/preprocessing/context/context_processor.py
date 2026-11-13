import re
from typing import Dict, Any, Optional, List

class ContextProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mask_demographics = self.config.get("mask_demographic_attributes", True)
        
        # Demographic attribute patterns to mask for ethics and bias prevention
        self.demographic_pattern = re.compile(
            r"\b(caste|religion|hindu|muslim|christian|sikh|jewish|black|white|male|female|gay|transgender|straight)\b",
            re.IGNORECASE
        )

    def process(self, context_obj: Optional[Any]) -> Dict[str, Any]:
        if not context_obj:
            return {
                "has_context": False,
                "formatted_context": "",
                "thread_length": 0,
                "sanitized_tokens": []
            }

        thread_history: List[str] = []
        platform = "unknown"
        post_type = "direct"

        if isinstance(context_obj, dict):
            thread_history = context_obj.get("thread_history") or []
            platform = context_obj.get("platform", "unknown")
            post_type = context_obj.get("post_type", "direct")
        elif hasattr(context_obj, "thread_history"):
            thread_history = getattr(context_obj, "thread_history") or []
            platform = getattr(context_obj, "platform", "unknown") or "unknown"
            post_type = getattr(context_obj, "post_type", "direct") or "direct"
        elif isinstance(context_obj, str):
            thread_history = [context_obj]

        sanitized_history = []
        for turn in thread_history:
            turn_clean = str(turn).strip()
            if self.mask_demographics:
                turn_clean = self.demographic_pattern.sub("[DEMOGRAPHIC_NEUTRAL]", turn_clean)
            if turn_clean:
                sanitized_history.append(turn_clean)

        formatted = " | ".join(sanitized_history)

        return {
            "has_context": len(sanitized_history) > 0,
            "formatted_context": formatted,
            "thread_length": len(sanitized_history),
            "platform": platform,
            "post_type": post_type,
            "masked_demographics": self.mask_demographics
        }

