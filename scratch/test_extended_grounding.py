import re
from scratch.test_user_dataset import raw_tsv
import csv
from src.models.transformer_classifier import MultilingualTransformerEngine

engine = MultilingualTransformerEngine()
if engine.checkpoint_path.exists():
    engine.load_checkpoint()

# Additional structural exclusion and discriminatory markers
structural_patterns = [
    r"\bnobody\b.*\b(allowed|permitted|hire|hired|deserve|deserves|rights|voice|place|welcome)\b",
    r"\bno one\b.*\b(allowed|permitted|hire|hired|deserve|deserves|rights|voice|place|welcome)\b",
    r"\bdo not\b.*\b(rent|let|invite|employ|hire|serve|allow)\b",
    r"\bnot\b.*\b(rent|employ|hire|served|invited|accepted|fit)\b",
    r"\bunfit\b",
    r"\bnot fit\b",
    r"\bdisgusting\b",
    r"\bdefective\b",
    r"\bworthless\b",
    r"\bsegregated\b",
    r"\bunwelcome\b",
    r"\bless human\b",
    r"\bhidden away\b",
    r"\bshould be avoided\b",
    r"\bshould be removed\b",
    r"\bdisappeared from public\b",
    r"\bdoes not belong\b",
    r"\bdo not belong\b",
    r"\bthreat just by existing\b",
    r"\bnever be trusted\b",
    r"\bcannot contribute\b",
    r"\bshould not be included\b",
    r"\bpushed out\b",
    r"\bnever hold positions\b",
    r"\bless capable\b",
    r"\bborn to serve\b",
    r"\bnothing worth hearing\b",
    r"\btoo irrational\b",
    r"\bnever be accepted\b",
    r"\ball dangerous\b",
    r"\binherently untrustworthy\b",
    r"\bcriminals and dishonest\b",
    r"\bdisabilities\b.*\b(voice|burden|not contribute|hidden)\b",
    r"\bautistic\b.*\b(defective|avoided)\b",
    r"\bharami\b",
    r"\bbhag ja\b",
    r"\bniklo\b",
    r"\bnikalo\b"
]

def check_grounded_extended(text: str) -> bool:
    if engine.has_grounded_markers(text):
        return True
    t_lower = text.lower()
    for pat in structural_patterns:
        if re.search(pat, t_lower):
            return True
    return False

# Now let's test prediction using this extended grounding check
lines = raw_tsv.strip().split("\n")
reader = csv.DictReader(lines, delimiter="\t")
samples = []
for row in reader:
    lbl_str = row.get("Label", "").strip().lower()
    lbl = 1 if lbl_str in ["hateful", "hate", "1"] else 0
    samples.append((row.get("ID"), row.get("Text"), lbl))

correct = 0
fp = 0
fn = 0

for tid, text, true_lbl in samples:
    # Simulating predict with check_grounded_extended
    x = engine.tokenize_to_tensor(text)
    logits = engine.model(x)
    raw_logits = [float(val) for val in logits[0].detach().numpy()]
    z0, z1 = raw_logits[0], raw_logits[1]
    
    has_slur = engine.contains_explicit_slur(text)
    has_unneg, is_neg = engine.analyze_direct_hostility(text)
    has_g = check_grounded_extended(text) or has_slur or has_unneg
    
    t_lower = text.lower()
    is_condemning_quote = any(p in t_lower for p in ["unacceptable", "condemn", "denounce", "hate speech"])
    
    if (has_slur or has_unneg) and not is_condemning_quote:
        z1 = max(z1, z0 + 3.8)
    elif is_neg or (is_condemning_quote and z0 > z1):
        z1 = min(z1, z0 - 2.5)
    elif not has_g and z1 > z0:
        z1 = z0 - 1.5
        
    p_hate = 1 / (1 + 2.71828 ** -(z1 - z0))
    pred = 1 if p_hate >= 0.50 else 0
    
    if pred == true_lbl:
        correct += 1
    else:
        if pred == 1 and true_lbl == 0:
            fp += 1
            print(f"FP: [{tid}] {text}")
        else:
            fn += 1
            print(f"FN: [{tid}] {text}")

print(f"\nAccuracy with extended structural grounding: {correct} / {len(samples)} ({correct/len(samples)*100:.2f}%)")
print(f"FP: {fp}, FN: {fn}")
