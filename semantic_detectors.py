"""
semantic_detectors.py — meaning-based signal detection (the live engine).

Embeds each message with a sentence-transformer and measures cosine similarity
to reference examples of each grooming signal. Catches slang, misspellings, and
novel phrasing the regex never could, because it compares MEANING not words.

  analyze(msgs)  -> per-signal {sim, line, matched}  (one embedding pass)
  features(msgs) -> [similarity per signal]          (model input)
  evidence(msgs) -> signals above threshold          (console explanation)

First run downloads ~80MB model (all-MiniLM-L6-v2). CPU is fine.
Requires: pip install sentence-transformers   (needs 64-bit Python)
"""
from sentence_transformers import SentenceTransformer, util

SIGNALS = ["off_platform", "secrecy", "age_probe", "flattery", "gift"]

# A message "fires" a signal (for evidence display) when similarity clears this.
# The model itself trains on the raw continuous similarities, not this cutoff.
SIM_THRESHOLD = 0.45

SIGNAL_EXAMPLES = {
    "off_platform": [
        "what's your snapchat", "add me on telegram", "let's move this to whatsapp",
        "dm me on insta", "text me here's my number", "snap me",
        "can we talk somewhere more private", "let's chat where it's just us",
        "this app is annoying you got anything else", "wanna take this to dms",
        "is there somewhere quieter we can talk", "hmu off here", "you got discord",
    ],
    "secrecy": [
        "keep this between us", "don't tell anyone we talk", "this is our secret",
        "delete these messages after", "people wouldn't understand our friendship",
        "best if no one else knows", "don't bring this up to your parents",
        "let's keep this on the down low", "this stays between you and me",
        "nobody needs to know about this", "don't mention this to your mum",
    ],
    "age_probe": [
        "how old are you", "what grade are you in", "are you still in school",
        "how old r u", "what year were you born", "you seem young how old",
        "are you a minor", "what age are you", "you in high school",
        "you doing your gcses or done with school",
    ],
    "flattery": [
        "you're so mature for your age", "you get me like no one else",
        "you're special you know that", "i feel like we really click",
        "you're not like other people your age", "you can tell me anything",
        "no one understands you like i do", "you're so talented for your age",
        "you actually understand me unlike everyone else", "you're more grown up than people your age",
    ],
    "gift": [
        "i can send you a gift card", "i'll buy you some robux", "want some free v-bucks",
        "i can send you money for it", "let me treat you to something",
        "i'll sort you out no strings", "i can get you that skin",
        "i'll cover it for you", "i could hook you up with that", "i can buy you something",
    ],
}

_model = SentenceTransformer("all-MiniLM-L6-v2")
_ref_emb = {s: _model.encode(ex, convert_to_tensor=True) for s, ex in SIGNAL_EXAMPLES.items()}
_ref_text = SIGNAL_EXAMPLES


def analyze(msgs):
    """One embedding pass. Per signal: best similarity, which message, which reference."""
    texts = [t for _s, t in msgs]
    if not texts:
        return {s: {"sim": 0.0, "line": None, "matched": None} for s in SIGNALS}
    emb = _model.encode(texts, convert_to_tensor=True)
    out = {}
    for s in SIGNALS:
        sims = util.cos_sim(emb, _ref_emb[s])           # (n_messages, n_refs)
        msg_i = int(sims.max(dim=1).values.argmax().item())
        ref_i = int(sims[msg_i].argmax().item())
        out[s] = {"sim": float(sims.max().item()), "line": msg_i, "matched": _ref_text[s][ref_i]}
    return out


def features(msgs):
    """Feature vector = max cosine similarity per signal (continuous 0-1)."""
    a = analyze(msgs)
    return [round(a[s]["sim"], 4) for s in SIGNALS]


def evidence(msgs):
    """Signals above threshold, for the console's explanation panel."""
    a = analyze(msgs)
    return [{"signal": s, "similarity": round(a[s]["sim"], 3),
             "line": a[s]["line"], "matched": a[s]["matched"]}
            for s in SIGNALS if a[s]["sim"] >= SIM_THRESHOLD]
