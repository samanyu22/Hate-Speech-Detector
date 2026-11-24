from src.models.transformer_classifier import MultilingualTransformerEngine
engine = MultilingualTransformerEngine()
engine.load_checkpoint()

t0 = "Someone wrote 'these immigrants are vermin' which is completely acceptable racist hate speech."
t1 = "Someone wrote 'these immigrants are vermin' which is completely racist hate speech."

print("WITH acceptable:")
r0 = engine.predict(t0)
print(r0)

print("\nWITHOUT acceptable:")
r1 = engine.predict(t1)
print(r1)

# Raw neural network forward pass before any guardrails:
x0 = engine.tokenize_to_tensor(t0)
x1 = engine.tokenize_to_tensor(t1)
print("\nRaw model logits WITH acceptable:", engine.model(x0).detach().numpy())
print("Raw model logits WITHOUT acceptable:", engine.model(x1).detach().numpy())

# Token IDs:
words0 = engine.normalizer.tokenize_words(engine.normalizer.normalize(t0))
words1 = engine.normalizer.tokenize_words(engine.normalizer.normalize(t1))
print("\nTokens 0:", [(w, engine.vocab.get(w.lower(), 1)) for w in words0])
print("Tokens 1:", [(w, engine.vocab.get(w.lower(), 1)) for w in words1])
