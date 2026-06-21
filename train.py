"""
Train + evaluate the hybrid model.

The model: logistic regression over the five rule-signal features. It learns
how much each signal should count, replacing the console's hand-set weights
with weights fit to data — while staying fully interpretable, because every
feature is a named signal you can read straight off the coefficients.

Outputs:
  - console metrics: precision / recall / F1 / confusion matrix on a held-out set
  - weights.json: learned coefficients + intercept + band cutoffs, for the console
"""
import json
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from detectors import SIGNALS, features

with open("dataset.json") as f:
    data = json.load(f)

X = np.array([features(d["msgs"]) for d in data])
y = np.array([d["label"] for d in data])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)

clf = LogisticRegression()
clf.fit(X_tr, y_tr)

proba = clf.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)

print("=" * 60)
print("LEARNED WEIGHTS  (hand-set value in the console, for comparison)")
print("=" * 60)
hand = {"off_platform": 22, "secrecy": 26, "age_probe": 16, "flattery": 16, "gift": 20}
for sig, w in sorted(zip(SIGNALS, clf.coef_[0]), key=lambda kv: -kv[1]):
    print(f"  {sig:<14} learned {w:+.2f}     (console: {hand[sig]})")
print(f"  {'(intercept)':<14} {clf.intercept_[0]:+.2f}")

print("\n" + "=" * 60)
print("EVALUATION  (held-out test set it never trained on)")
print("=" * 60)
print(classification_report(y_te, pred, target_names=["safe", "grooming"], digits=3))
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print("Confusion matrix:")
print(f"                 predicted safe   predicted grooming")
print(f"  actual safe          {tn:3d}                {fp:3d}        <- {fp} false positives")
print(f"  actual grooming      {fn:3d}                {tp:3d}        <- {fn} false negatives (missed)")
print(f"\n  ROC-AUC: {roc_auc_score(y_te, proba):.3f}")

# Band cutoffs on the predicted probability. 'review' is deliberately wide so
# uncertain cases route to a human instead of being cleared or auto-flagged.
BANDS = {"review": 0.35, "high": 0.70}

weights = {
    "intercept": clf.intercept_[0],
    "weights": dict(zip(SIGNALS, clf.coef_[0].tolist())),
    "bands": BANDS,
    "note": "console computes p = sigmoid(intercept + sum(w_i * signal_i)); maps p to band",
}
with open("weights.json", "w") as f:
    json.dump(weights, f, indent=2)
print("\nwrote weights.json — drop these learned weights into the console")
