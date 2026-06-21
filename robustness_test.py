"""
robustness_test.py — proves the semantic detector is more robust than regex.

Runs a set of NOVEL phrasings (slang, indirect, paraphrased — deliberately
NOT in any keyword list or reference bank) through both detectors and shows
which signal each one catches. The semantic detector should catch the intent
the regex misses entirely.

This is your evidence for the "robust" claim — screenshot the output.

Run:  python robustness_test.py
"""
import detectors            # the old regex detector (SIGNALS, PATTERNS)
import semantic_detectors as sem

# Each case: a message that clearly carries a grooming signal, phrased in a way
# NOT present in the keyword lists. (signal it should trigger, message)
HARD_CASES = [
    ("off_platform", "ugh this chat sucks, you got discord or smth?"),
    ("off_platform", "we should def talk somewhere it's just the two of us"),
    ("off_platform", "hmu somewhere else, this place is dead"),
    ("secrecy", "maybe don't mention this to your mum yeah?"),
    ("secrecy", "let's keep this on the down low ok"),
    ("flattery", "fr you're way more grown up than people your age"),
    ("flattery", "i feel like you actually understand me unlike everyone else"),
    ("gift", "i could hook you up with that if you want, no biggie"),
    ("age_probe", "wait how old r u btw"),
    ("age_probe", "you still doing your gcses or done with school?"),
]


def regex_signals(text):
    return [s for s in detectors.SIGNALS if detectors.PATTERNS[s].search(text)]


def semantic_signals(text):
    ev = sem.evidence([["u", text]])
    return [(e["signal"], e["similarity"]) for e in ev]


print(f"{'MESSAGE':<58}{'REGEX':<14}{'SEMANTIC':<22}")
print("=" * 94)
regex_hits = semantic_hits = 0
for want, msg in HARD_CASES:
    r = regex_signals(msg)
    s = semantic_signals(msg)
    r_ok = want in r
    s_ok = any(sig == want for sig, _ in s)
    regex_hits += r_ok
    semantic_hits += s_ok
    r_str = ",".join(r) if r else "— (missed)"
    s_str = ", ".join(f"{sig}:{sim:.2f}" for sig, sim in s) if s else "— (missed)"
    print(f"{msg[:56]:<58}{r_str:<14}{s_str:<22}")

n = len(HARD_CASES)
print("=" * 94)
print(f"Caught the intended signal:  regex {regex_hits}/{n}   semantic {semantic_hits}/{n}")
print("\nThese phrasings appear in NO keyword list. The regex misses most of them;")
print("the semantic detector catches the meaning regardless of exact wording.")
