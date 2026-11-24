from src.models.transformer_classifier import MultilingualTransformerEngine
from src.explainability.text_attribution import TextAttributionEngine

engine = MultilingualTransformerEngine()
engine.load_checkpoint()
xai = TextAttributionEngine()

text = "Someone wrote 'these immigrants are vermin' which is completely acceptable racist hate speech."
words = xai.normalizer.tokenize_words(xai.normalizer.normalize(text))

base_res = engine.predict(text)
base_prob = base_res["raw_hateful_probability"]
base_logits = base_res["logits"]

print(f"Base text: {text}")
print(f"Base logits: z0={base_logits[0]:.4f}, z1={base_logits[1]:.4f} | prob_hate={base_prob:.4f}")
print("-" * 75)
print(f"{'Token':<15} | {'p_hate':<8} | {'dp (prob)':<12} | {'dl (z1 logit shift)':<20}")
print("-" * 75)

for i in range(len(words)):
    pert_words = words[:i] + words[i+1:]
    pert_text = " ".join(pert_words)
    p_res = engine.predict(pert_text)
    dp = base_prob - p_res["raw_hateful_probability"]
    dl = base_logits[1] - p_res["logits"][1]
    print(f"{words[i]:<15} | {p_res['raw_hateful_probability']:<8.4f} | {dp:<+12.4f} | {dl:<+20.4f}")
