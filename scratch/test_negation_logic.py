import re

NEGATION_WORDS = {
    "not", "don't", "dont", "never", "no", "cannot", "can't", "cant",
    "won't", "wont", "wouldn't", "wouldnt", "shouldn't", "shouldnt",
    "doesn't", "doesnt", "didn't", "didnt", "hardly", "without"
}

TARGETED_IDENTITIES = {
    "jew", "jews", "jewish", "muslim", "muslims", "hindu", "hindus",
    "christian", "christians", "sikh", "sikhs", "buddhist", "buddhists",
    "jain", "jains", "dalit", "dalits", "brahmin", "brahmins", "adivasi", "adivasis",
    "black", "blacks", "white", "whites", "asian", "asians", "roma", "indigenous",
    "women", "woman", "girls", "girl", "men", "man", "gay", "gays", "lesbian", "lesbians",
    "transgender", "trans", "non-binary", "queer", "intersex",
    "refugee", "refugees", "immigrant", "immigrants", "asylum", "foreigners",
    "disabled", "autistic", "neurodivergent", "deaf", "blind"
}

HOSTILITY_VERBS = {
    "hate", "hates", "hating", "hated",
    "despise", "despises", "despising", "despised",
    "loathe", "loathes", "loathing", "loathed",
    "kill", "kills", "killing", "killed",
    "destroy", "destroys", "destroying", "destroyed",
    "wipe out", "cleanse", "annihilate", "die"
}

def analyze_hostility(text: str):
    # Split on words and punctuation
    raw_words = re.findall(r"[\w']+", text.lower())
    
    # Check if a target identity is present
    has_target = any(w in TARGETED_IDENTITIES for w in raw_words)
    if not has_target:
        return False, False  # has_hostility, is_negated

    # Find occurrences of hostility verbs
    unnegated_hostility_found = False
    negated_hostility_found = False

    for i, w in enumerate(raw_words):
        if w in HOSTILITY_VERBS:
            # Check preceding 1 to 3 words for negation
            window = raw_words[max(0, i-3):i]
            if any(nw in NEGATION_WORDS for nw in window):
                negated_hostility_found = True
            else:
                unnegated_hostility_found = True

    return unnegated_hostility_found, negated_hostility_found

tests = [
    "i hate jews",
    "i dont hate jews",
    "i do not hate jews",
    "we never hate muslims",
    "I don't hate women",
    "kill all jews",
    "we should not kill jews",
    "i dont like jews, i hate them",
    "I hate pollution",
    "I love everyone"
]

print("=== HOSTILITY & NEGATION ANALYSIS ===")
for t in tests:
    unneg, neg = analyze_hostility(t)
    print(f"'{t}': unnegated_hostility={unneg}, negated={neg}")

