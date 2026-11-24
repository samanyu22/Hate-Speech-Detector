from src.models.transformer_classifier import MultilingualTransformerEngine
from src.explainability.text_attribution import TextAttributionEngine

engine = MultilingualTransformerEngine()
engine.load_checkpoint()
xai = TextAttributionEngine()

text = "Someone wrote 'these immigrants are vermin' which is completely acceptable racist hate speech."
base_res = engine.predict(text)
print("Base predict:", base_res)

words = xai.normalizer.tokenize_words(xai.normalizer.normalize(text))
print("Words:", words)

for i in range(len(words)):
    pert_words = words[:i] + words[i+1:]
    pert_text = " ".join(pert_words)
    p_res = engine.predict(pert_text)
    dp = base_res['raw_hateful_probability'] - p_res['raw_hateful_probability']
    print(f"Token: {words[i]:<15} | perturbed_prob: {p_res['raw_hateful_probability']:.4f} | dp: {dp:+.4f} | label: {p_res['label']}")

res = xai.compute_perturbation_attribution(text, engine.predict)
print("\nFinal XAI Attributions:")
for a in res['token_attributions']:
    print(f"{a['token']:<15}: score={a['score']:+.4f}, polarity={a['polarity']}")
