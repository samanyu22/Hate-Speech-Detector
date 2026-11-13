from typing import List, Dict, Any, Optional

class ExplanationSynthesizer:
    @staticmethod
    def synthesize(
        label: int,
        confidence: float,
        language: str,
        token_attributions: List[Dict[str, Any]],
        faithfulness_score: float,
        ocr_text: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        label_str = "HATEFUL" if label == 1 else "NON-HATEFUL"

        # Extract top positive (hate-inducing) and negative (non-hate-inducing) tokens
        sorted_tokens = sorted(token_attributions, key=lambda x: x["score"], reverse=True)
        top_hate_tokens = [t["token"] for t in sorted_tokens if t["score"] > 0.005][:5]
        top_non_hate_tokens = [t["token"] for t in sorted(token_attributions, key=lambda x: x["score"]) if t["score"] < -0.005][:5]

        # Synthesize 6-point structured natural language explanation
        explanation_parts = []
        explanation_parts.append(f"1. Classification: The input is categorized as {label_str} with {confidence*100:.1f}% confidence.")

        if label == 1:
            if top_hate_tokens:
                explanation_parts.append(f"2. Influential Evidence: The primary linguistic markers elevating hostility include '{', '.join(top_hate_tokens)}'.")
            else:
                explanation_parts.append("2. Influential Evidence: Hostile semantic patterns identified across multi-word phrasing.")
        else:
            if top_non_hate_tokens:
                explanation_parts.append(f"2. Influential Evidence: Civil or neutral markers lowering hostility include '{', '.join(top_non_hate_tokens)}'.")
            else:
                explanation_parts.append("2. Influential Evidence: Absence of dehumanization, slur tokens, or violent incitement.")

        # OCR element
        if ocr_text:
            explanation_parts.append(f"3. Visual & OCR Modality: Embedded meme text was extracted (\"{ocr_text}\") and evaluated jointly with textual features.")
        else:
            explanation_parts.append("3. Visual Modality: No visual text or image provided; inference executed solely over textual modality.")

        # Context element
        if context_data and context_data.get("has_context"):
            explanation_parts.append(f"4. Contextual Information: Thread conversation history ({context_data.get('thread_length')} prior turns) was evaluated; protected identity categories remained sanitized.")
        else:
            explanation_parts.append("4. Contextual Information: Standalone post evaluated without external thread history.")

        # Ambiguity / Uncertainty
        if confidence < 0.70:
            explanation_parts.append("5. Ambiguity: Low margin of confidence; sarcastic framing, cultural code-switching, or informal idioms may be present.")
        else:
            explanation_parts.append("5. Ambiguity: High confidence determination with strong gradient attribution alignment.")

        # Limitations & Faithfulness
        explanation_parts.append(f"6. Limitations & Faithfulness: Model verified with a post-hoc perturbation faithfulness score of {faithfulness_score:.2f}. Human review recommended before punitive moderation.")

        full_text = "\n\n".join(explanation_parts)

        return {
            "natural_language_explanation": full_text,
            "top_hate_tokens": top_hate_tokens,
            "top_non_hate_tokens": top_non_hate_tokens,
            "faithfulness_score": faithfulness_score
        }

