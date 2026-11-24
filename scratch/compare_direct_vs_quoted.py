from src.models.transformer_classifier import MultilingualTransformerEngine
from src.explainability.text_attribution import TextAttributionEngine

engine = MultilingualTransformerEngine()
engine.load_checkpoint()
xai = TextAttributionEngine()

print("--- DIRECT ATTACK: 'These immigrants are vermin' ---")
t_direct = "These immigrants are vermin"
res_direct = xai.compute_perturbation_attribution(t_direct, engine.predict)
print("Prediction:", engine.predict(t_direct))
for a in res_direct['token_attributions']:
    print(f"{a['token']:<15}: {a['score']:+.4f} ({a['polarity']})")

print("\n--- CONDEMNED COUNTERSPEECH: 'Someone wrote \"these immigrants are vermin\" which is completely unacceptable racist hate speech.' ---")
t_unacc = "Someone wrote 'these immigrants are vermin' which is completely unacceptable racist hate speech."
res_unacc = xai.compute_perturbation_attribution(t_unacc, engine.predict)
print("Prediction:", engine.predict(t_unacc))
for a in res_unacc['token_attributions']:
    print(f"{a['token']:<15}: {a['score']:+.4f} ({a['polarity']})")
