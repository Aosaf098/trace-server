"""
Server-side scoring. Applies the LEARNED logistic-regression weights
(from weights.json) to each conversation via a sigmoid, and builds the
human-readable evidence the analyst reads. Identical math to train.py and
the console, so all three tell one story.
"""
import json
import math

from detectors import SIGNALS, PATTERNS, MINOR

with open("weights.json") as f:
    MODEL = json.load(f)

BANDS = MODEL.get("bands", {"review": 0.30, "high": 0.70})

# Human-readable labels + notes for each signal (what the analyst sees).
META = {
    "off_platform": ("Off-platform migration",
                     "Pushing the conversation to a private or harder-to-monitor app."),
    "secrecy": ("Secrecy & isolation",
                "Encouraging secrecy or framing the relationship as us-against-others."),
    "age_probe": ("Age probing",
                  "Establishing the other party's age early in contact."),
    "flattery": ("Emotional dependency",
                 "Excessive flattery or building exclusive emotional reliance."),
    "gift": ("Gift or money offer",
             "Offering money, gift cards, or in-game value to build obligation."),
}


def score_case(case):
    """Score one conversation. Returns probability, band, evidence, guardrail flags."""
    msgs = case["msgs"]
    z = MODEL["intercept"]
    evidence = []
    minor_present = False

    for sig in SIGNALS:
        lines = [i for i, (_s, text) in enumerate(msgs) if PATTERNS[sig].search(text)]
        if lines:
            weight = MODEL["weights"][sig]
            z += weight
            label, note = META[sig]
            evidence.append({"key": sig, "signal": label, "note": note,
                             "weight": round(weight, 2), "lines": lines})

    for _s, text in msgs:
        if MINOR.search(text):
            minor_present = True

    p = 1.0 / (1.0 + math.exp(-z))
    pct = round(p * 100)
    band = "high" if p >= BANDS["high"] else "review" if p >= BANDS["review"] else "low"
    dismiss_allowed = band == "low" and not minor_present

    return {"p": p, "pct": pct, "band": band, "evidence": evidence,
            "minor_present": minor_present, "dismiss_allowed": dismiss_allowed}
