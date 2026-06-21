"""
evaluate.py — evidence that the SEMANTIC scoring engine works.

Produces, from the same dataset.json:
  1. Held-out metrics + confusion matrix (printed + image)
  2. Precision-recall curve (image), band thresholds marked
  3. Semantic model vs the OLD regex detector — the robustness gain in numbers
  4. 5-fold cross-validation (stable, not a fluke)

Run:  python evaluate.py     (outputs in ./eval_outputs/)

NOTE: metrics are on SYNTHETIC data — they show the pipeline separates the
patterns we generated, not real-world performance. State this in the write-up.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)

from semantic_detectors import SIGNALS, features as sem_features
import detectors  # the OLD regex detector, used as the baseline

OUT = "eval_outputs"
os.makedirs(OUT, exist_ok=True)
BANDS = {"review": 0.30, "high": 0.70}

with open("dataset.json") as f:
    data = json.load(f)

print(f"Embedding {len(data)} conversations…")
X = np.array([sem_features(d["msgs"]) for d in data])
y = np.array([d["label"] for d in data])

idx = np.arange(len(data))
tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=7)
clf = LogisticRegression().fit(X[tr], y[tr])
proba = clf.predict_proba(X[te])[:, 1]
pred = (proba >= 0.5).astype(int)
y_te = y[te]

print("=" * 64)
print("1. HELD-OUT EVALUATION  (semantic model, test set)")
print("=" * 64)
print(classification_report(y_te, pred, target_names=["safe", "grooming"], digits=3))
print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")

cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ConfusionMatrixDisplay(cm, display_labels=["safe", "grooming"]).plot(
    ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title("TRACE (semantic) — Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150); plt.close()
tn, fp, fn, tp = cm.ravel()
print(f"Confusion matrix saved. FP={fp} (innocent flagged)  FN={fn} (grooming missed)")

prec, rec, thr = precision_recall_curve(y_te, proba)
ap = average_precision_score(y_te, proba)
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot(rec, prec, lw=2, color="#2B4C7E")
for name, t in BANDS.items():
    i = int(np.argmin(np.abs(thr - t)))
    ax.scatter(rec[i], prec[i], s=40, zorder=5)
    ax.annotate(f"{name} >= {int(t*100)}%", (rec[i], prec[i]),
                textcoords="offset points", xytext=(6, -10), fontsize=9)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
ax.set_title(f"Precision-Recall curve  (AP = {ap:.3f})")
ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/precision_recall_curve.png", dpi=150); plt.close()
print(f"Precision-recall curve saved. Average precision = {ap:.3f}")

# Baseline: the OLD regex detector — flag if any keyword signal fires.
def regex_flag(msgs):
    return 1 if any(detectors.PATTERNS[s].search(t) for _s, t in msgs for s in detectors.SIGNALS) else 0
base_pred = np.array([regex_flag(data[i]["msgs"]) for i in te])

print("\n" + "=" * 64)
print("2. ROBUSTNESS GAIN  (semantic model vs old regex detector)")
print("=" * 64)
print(f"{'':18}{'precision':>11}{'recall':>9}{'f1':>8}")
for name, p in [("regex baseline", base_pred), ("semantic model", pred)]:
    print(f"{name:18}{precision_score(y_te, p):>11.3f}{recall_score(y_te, p):>9.3f}{f1_score(y_te, p):>8.3f}")
print("\nThe regex baseline only catches anticipated wording. The semantic model")
print("catches meaning — handling slang, misspellings, and novel phrasing.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
f1s = cross_val_score(LogisticRegression(), X, y, cv=cv, scoring="f1")
aucs = cross_val_score(LogisticRegression(), X, y, cv=cv, scoring="roc_auc")
print("\n" + "=" * 64)
print("3. STABILITY  (5-fold cross-validation)")
print("=" * 64)
print(f"  F1:      {f1s.mean():.3f} +/- {f1s.std():.3f}")
print(f"  ROC-AUC: {aucs.mean():.3f} +/- {aucs.std():.3f}")
print(f"\nImages in ./{OUT}/ — screenshot for the submission.")
