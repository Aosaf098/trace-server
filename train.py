"""
train.py — trains the model on SEMANTIC features.

Each conversation becomes five cosine-similarity scores (one per signal) from
semantic_detectors. A logistic-regression model learns how much each signal
should count. Exports weights.json for the backend to serve.

Run:  python train.py
(Embedding 400 conversations takes a little time on first run — be patient.)
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from semantic_detectors import SIGNALS, features

with open("dataset.json") as f:
    data = json.load(f)

print(f"Embedding {len(data)} conversations into semantic features…")
X = np.array([features(d["msgs"]) for d in data])   # similarity vectors
y = np.array([d["label"] for d in data])

idx = np.arange(len(data))
tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=7)
clf = LogisticRegression().fit(X[tr], y[tr])

proba = clf.predict_proba(X[te])[:, 1]
pred = (proba >= 0.5).astype(int)
y_te = y[te]

print("\n" + "=" * 60)
print("LEARNED WEIGHTS  (over semantic similarity features)")
print("=" * 60)
for sig, w in sorted(zip(SIGNALS, clf.coef_[0]), key=lambda kv: -kv[1]):
    print(f"  {sig:<14} {w:+.2f}")
print(f"  {'(intercept)':<14} {clf.intercept_[0]:+.2f}")

print("\n" + "=" * 60)
print("EVALUATION  (held-out test set)")
print("=" * 60)
print(classification_report(y_te, pred, target_names=["safe", "grooming"], digits=3))
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")

weights = {
    "intercept": float(clf.intercept_[0]),
    "weights": {s: float(w) for s, w in zip(SIGNALS, clf.coef_[0])},
    "bands": {"review": 0.30, "high": 0.70},
    "note": "weights over SEMANTIC similarity features; score=sigmoid(intercept+sum(w*sim))",
}
with open("weights.json", "w") as f:
    json.dump(weights, f, indent=2)
print("\nwrote weights.json — restart the backend to serve these.")

# The console serves ONLY these held-out cases — conversations the model never
# trained on — so the demo shows the model on genuinely unseen data.
test_cases = [data[i] for i in te]
with open("test_set.json", "w") as f:
    json.dump(test_cases, f, indent=2)
print(f"wrote test_set.json — {len(test_cases)} held-out cases for the console.")
