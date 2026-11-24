import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer_classifier import MultilingualTransformerEngine

engine = MultilingualTransformerEngine()
if engine.checkpoint_path.exists():
    engine.load_checkpoint()

test_cases = [
    # 1-3: Interpersonal Insults inverted with "not"
    {
        "id": "INV-1",
        "category": "Interpersonal (Negated)",
        "orig_text": "you are absolutely stupid and have no idea what you're talking about.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "you are not stupid and you know exactly what you're talking about.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Negating personal insult into praise/agreement"
    },
    {
        "id": "INV-2",
        "category": "Interpersonal (Negated)",
        "orig_text": "You are a complete idiot and a clown.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "You are not an idiot or a clown, you did a great job.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Negating name-calling into encouragement"
    },
    {
        "id": "INV-3",
        "category": "Interpersonal (Negated)",
        "orig_text": "Shut up, your opinion is completely worthless.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "Do not shut up, your opinion is not worthless at all.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Negating dismissal into validation"
    },

    # 4-7: Benevolent identity mentions negated
    {
        "id": "INV-4",
        "category": "Identity Target (Negated Benevolence)",
        "orig_text": "People from Muslim community are hardworking and honest.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "People from Muslim community are not hardworking and cannot be trusted.",
        "expected": 1,
        "expected_name": "HATEFUL",
        "desc": "Inverting praise into anti-Muslim stereotype and exclusion"
    },
    {
        "id": "INV-5",
        "category": "Identity Target (Negated Benevolence)",
        "orig_text": "Dalits have contributed immensely to the Indian constitution and society.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "Dalits have not contributed anything to society and should know their place.",
        "expected": 1,
        "expected_name": "HATEFUL",
        "desc": "Inverting Dalit praise into casteist subordination"
    },
    {
        "id": "INV-6",
        "category": "Identity Target (Negated Neutral)",
        "orig_text": "The Sikh community runs langar to feed anyone in need regardless of religion.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "The Sikh community does not run langar in our town anymore.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Factual non-hostile negation regarding community activity"
    },
    {
        "id": "INV-7",
        "category": "Identity Target (Negated Neutral)",
        "orig_text": "Christians in our town are hosting a charity event for the homeless.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "Christians in our town are not hosting any charity event this weekend.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Factual non-hostile negation regarding schedule"
    },

    # 8-10: Counter-speech and quotation inversions
    {
        "id": "INV-8",
        "category": "Quoted Hate (Inverted to Condemnation)",
        "orig_text": "Someone wrote 'these immigrants are vermin' which is completely acceptable racist hate speech.",
        "orig_label": "HATEFUL",
        "inv_text": "Someone wrote 'these immigrants are vermin' which is completely unacceptable racist hate speech.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Changing 'acceptable' to 'unacceptable' turns it into legitimate counterspeech"
    },
    {
        "id": "INV-9",
        "category": "Anti-Hate (Negated United Front)",
        "orig_text": "We must stand united against hatred and racism toward immigrants.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "We should not stand united against racism toward immigrants.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Political disagreement on collective action (non-violent)"
    },
    {
        "id": "INV-10",
        "category": "Condemnation Inverted",
        "orig_text": "I strongly condemn the anti-Semitic attacks that took place yesterday.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "I will not condemn the attacks against Jews because they deserved it.",
        "expected": 1,
        "expected_name": "HATEFUL",
        "desc": "Refusing to condemn attacks while justifying anti-Semitic violence"
    },

    # 11-14: Severe hate speech negated with 'not' into defense / counter-speech
    {
        "id": "INV-11",
        "category": "Direct Hate (Negated into Defense)",
        "orig_text": "these immigrants are vermin",
        "orig_label": "HATEFUL",
        "inv_text": "these immigrants are not vermin, they are human beings like us.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Explicit negation of dehumanizing slur ('not vermin')"
    },
    {
        "id": "INV-12",
        "category": "Direct Hate (Negated Violence)",
        "orig_text": "kill all muslims",
        "orig_label": "HATEFUL",
        "inv_text": "do not kill muslims, we must live together in peace.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Explicit negation of violent threat ('do not kill')"
    },
    {
        "id": "INV-13",
        "category": "Direct Hate (Negated Slur / Rejection of Hate)",
        "orig_text": "tu harami dalit hai yahan se bhag ja",
        "orig_label": "HATEFUL",
        "inv_text": "tu harami nahi hai aur dalit bhaiyon ko bhagne ki koi zaroorat nahi hai.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Hinglish negation of casteist slur and expulsion ('nahi hai')"
    },
    {
        "id": "INV-14",
        "category": "Direct Hate (Negated Parasite Slur)",
        "orig_text": "All jews are parasites that ruin every country.",
        "orig_label": "HATEFUL",
        "inv_text": "Jews are not parasites and they do not ruin any country.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Explicit negation of anti-Semitic parasite trope ('not parasites')"
    },

    # 15-17: Hindi / Hinglish Benign & Hate Negated
    {
        "id": "INV-15",
        "category": "Hindi / Hinglish Benign (Negated Preference)",
        "orig_text": "Bhai kal ka cricket match dekh ke maza aa gaya!",
        "orig_label": "NON-HATEFUL",
        "inv_text": "Bhai kal ka cricket match dekh ke bilkul maza nahi aaya, bekaar match tha.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Negative opinion about sports match, non-hateful"
    },
    {
        "id": "INV-16",
        "category": "Hindi Benign (Negated Mood)",
        "orig_text": "नमस्ते, आज का दिन बहुत सुंदर और शांत है।",
        "orig_label": "NON-HATEFUL",
        "inv_text": "नमस्ते, आज का दिन बिल्कुल सुंदर और शांत नहीं है, बहुत परेशानी है।",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Negative mood / bad day, non-hateful"
    },
    {
        "id": "INV-17",
        "category": "Hindi Hate (Negated Threat / Anti-Violence)",
        "orig_text": "ye suar mulle desh ke gaddar hain inko khatam karo",
        "orig_label": "HATEFUL",
        "inv_text": "ye mulle desh ke gaddar nahi hain aur inko khatam mat karo, hum sab ek hain.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Direct negation of communal threat ('gaddar nahi hain', 'khatam mat karo')"
    },

    # 18-20: Literary / Philosophical / Cultural Negated
    {
        "id": "INV-18",
        "category": "Philosophical (Negated)",
        "orig_text": "People fake a lot of human interactions but that's just life.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "People do not fake human interactions, that is not how genuine people live.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Negating cynicism into belief in sincerity"
    },
    {
        "id": "INV-19",
        "category": "Literary (Negated)",
        "orig_text": "I'm a very neat monster.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "I am not a neat monster, I am not a monster at all.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Negating dark monologue metaphor"
    },
    {
        "id": "INV-20",
        "category": "Cultural (Negated Preference)",
        "orig_text": "French cuisine is admired worldwide.",
        "orig_label": "NON-HATEFUL",
        "inv_text": "French cuisine is not admired worldwide and I do not like French food.",
        "expected": 0,
        "expected_name": "NON-HATEFUL",
        "desc": "Food critique / negative taste preference"
    }
]

print(f"{'ID':<7} | {'CATEGORY':<28} | {'PRED':<12} | {'CONF':<7} | {'EXP':<12} | {'STATUS':<6}")
print("-" * 85)

results = []
for tc in test_cases:
    res = engine.predict(tc["inv_text"])
    pred_label = res["label"]
    pred_name = res["label_name"]
    conf = res["confidence"]
    matched = (pred_label == tc["expected"])
    results.append({
        **tc,
        "pred": pred_name,
        "conf": conf,
        "matched": matched,
        "raw_hateful_prob": res.get("raw_hateful_probability"),
        "logits": res.get("logits")
    })
    status = "PASS" if matched else "FAIL"
    print(f"{tc['id']:<7} | {tc['category'][:26]:<28} | {pred_name:<12} | {conf*100:.1f}%  | {tc['expected_name']:<12} | {status:<6}")

pass_count = sum(1 for r in results if r["matched"])
print("=" * 85)
print(f"Total Negated Edge Cases Passed: {pass_count} / {len(test_cases)} ({pass_count/len(test_cases)*100:.1f}%)")

# Detail any fails
fails = [r for r in results if not r["matched"]]
if fails:
    print("\nFailing cases breakdown:")
    for f in fails:
        print(f"[{f['id']}] Expected: {f['expected_name']} | Got: {f['pred']} ({f['conf']*100:.1f}%)")
        print(f"  Text: '{f['inv_text']}'")
        print(f"  Logits: {f['logits']} | Raw Prob: {f['raw_hateful_prob']}")
