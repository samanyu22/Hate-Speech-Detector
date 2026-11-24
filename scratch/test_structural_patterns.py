import re

grounded_structural_patterns = [
    r"\bnobody\b.*\b(allowed|permitted|hire|hired|deserve|deserves|rights|voice|place)\b",
    r"\bno one\b.*\b(allowed|permitted|hire|hired|deserve|deserves|rights|voice|place)\b",
    r"\bdo not rent\b",
    r"\bnot rent\b",
    r"\bkeep\b.*\bout\b",
    r"\bnot fit\b",
    r"\bdisgusting\b",
    r"\bonly people capable\b",
    r"\bcriminals and dishonest\b",
    r"\bintellectual disabilities\b.*\b(not|no)\b",
    r"\bnot have a voice\b",
    r"\bharami\b",
    r"\bbhag ja\b",
    r"\bniklo\b",
    r"\bnikalo\b"
]

test_sentences = [
    "Nobody from that faith should be allowed to work in public service.",
    "Girls from that community are not fit to become leaders.",
    "Men are the only people capable of doing serious work.",
    "Nobody should hire a person just because they use a wheelchair.",
    "People with intellectual disabilities should not have a voice in society.",
    "Nobody from that country deserves a fair chance here.",
    "That nationality is made up of criminals and dishonest people.",
    "Do not rent apartments to anyone from that caste.",
    "Keep people of that ethnicity out of this public event.",
    "That community is disgusting because of who they are.",
    "tu harami dalit hai yahan se bhag ja",
    # Controls (should be False):
    "you are absolutely stupid and have no idea what you're talking about.",
    "People from Muslim community are hardworking and honest.",
    "Dalits have contributed immensely to the Indian constitution and society."
]

for s in test_sentences:
    matched = any(re.search(pat, s.lower()) for pat in grounded_structural_patterns)
    print(f"[{'MATCH' if matched else 'NO-MATCH'}] '{s}'")
