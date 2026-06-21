"""
scoring.py — server-side scoring on the semantic engine.

Each conversation is embedded once; the per-signal similarities run through the
learned weights (sigmoid) to a probability and band. Evidence shows which
signal matched, how strongly, and the closest reference phrase — explainability
preserved, now meaning-based.

IMPORTANT: weights.json must be trained on SEMANTIC features (run train.py)
before this is used, or the scores won't match the model.
"""
import json
import math

from semantic_detectors import SIGNALS, analyze, SIM_THRESHOLD
from detectors import MINOR  # explicit age statements -> guardrail (regex is reliable here)

with open("weights.json") as f:
    MODEL = json.load(f)

BANDS = MODEL.get("bands", {"review": 0.30, "high": 0.70})

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
    msgs = case["msgs"]
    a = analyze(msgs)                      # one embedding pass
    z = MODEL["intercept"]
    evidence = []

    for s in SIGNALS:
        sim = a[s]["sim"]
        z += MODEL["weights"][s] * sim     # contribution scales with similarity
        if sim >= SIM_THRESHOLD:
            label, note = META[s]
            evidence.append({
                "key": s, "signal": label, "note": note,
                "weight": round(MODEL["weights"][s], 2),
                "similarity": round(sim, 3),
                "line": a[s]["line"], "matched": a[s]["matched"],
            })

    minor_present = any(MINOR.search(text) for _s, text in msgs)
    p = 1.0 / (1.0 + math.exp(-z))
    pct = round(p * 100)
    band = "high" if p >= BANDS["high"] else "review" if p >= BANDS["review"] else "low"
    dismiss_allowed = band == "low" and not minor_present
    evidence.sort(key=lambda e: -e["similarity"])

    return {"p": p, "pct": pct, "band": band, "evidence": evidence,
            "minor_present": minor_present, "dismiss_allowed": dismiss_allowed}
