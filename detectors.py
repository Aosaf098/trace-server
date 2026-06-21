"""
Signal detectors — the single shared source of truth for TRACE.

Used by:
  - app.py / scoring.py  (serving: which lines fired, for evidence + scoring)
  - train.py             (training: conversation -> feature vector)
  - evaluate.py          (evaluation: same features, held-out testing)

Each detector is a regex over message text. `detect` returns the message
indices each signal fired on; `features` turns a conversation into the binary
feature vector the model learns from. Same signals everywhere — so the demo,
the training, and the evaluation all tell one story.
"""
import re

# The five scored signals. Order is fixed — it defines the feature vector.
SIGNALS = ["off_platform", "secrecy", "age_probe", "flattery", "gift"]

PATTERNS = {
    "off_platform": re.compile(
        r"\b(snapchat|snap|whatsapp|telegram|kik|insta|instagram dm|"
        r"move (this )?to|dm me|text me|my number|add me on)\b", re.I),
    "secrecy": re.compile(
        r"\b(don'?t tell|between us|our (space|secret|friendship)|keep it between|"
        r"wouldn'?t (get|understand)|delete (these|the|our) messages|"
        r"don'?t mention|just between)\b", re.I),
    "age_probe": re.compile(
        r"\b(how old (are|r) (you|u)|what grade|are you \d{1,2})\b", re.I),
    "flattery": re.compile(
        r"\b(mature for|more mature|you get me|only one (who|that)|"
        r"talk to you about anything|so talented for your age|tell me anything)\b", re.I),
    "gift": re.compile(
        r"\b(gift card|robux|v-?bucks|send you money|buy you|paypal|"
        r"cashapp|free skin|i can send you)\b", re.I),
}

# Context flag — NOT a scored feature. Drives the mandatory-review guardrail
# (a minor being present doesn't predict grooming; it raises the duty of care).
MINOR = re.compile(
    r"\b(i'?m 1[0-7]\b|im 1[0-7]\b|in (6th|7th|8th|9th|10th) grade|turning 1[0-7])\b", re.I)


def detect(msgs):
    """Run every detector over a conversation (list of [sender, text]).

    Returns: {'fired': {signal -> [message indices]}, 'minor_present': bool}
    """
    fired = {s: [] for s in SIGNALS}
    minor_present = False
    for i, (_sender, text) in enumerate(msgs):
        for s in SIGNALS:
            if PATTERNS[s].search(text):
                fired[s].append(i)
        if MINOR.search(text):
            minor_present = True
    return {"fired": fired, "minor_present": minor_present}


def features(msgs):
    """Convert a conversation into the model's feature vector.

    Binary presence of each signal, in SIGNALS order. Interpretable by design:
    every feature is a named, human-readable risk signal.
    """
    d = detect(msgs)
    return [1 if d["fired"][s] else 0 for s in SIGNALS]
