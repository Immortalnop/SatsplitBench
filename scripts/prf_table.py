"""Threshold-based classification metrics for presence leakage (paper Table: precision, recall, F1, confusion matrix).

For each input (4 audited encoders + 2 controls, full size, split seed 0), the Phase 4b attacker with the best
validation AUC is taken (Phase 4 probes saved no validation scores, so they are not candidates here). The decision
threshold is the one that maximises F1 on the validation set; precision, recall, F1, accuracy and the confusion
matrix are then computed once on the test set.

Input : results/preds/phase4b/seed0/<encoder>/full/leakage_<attacker>.npz  (scores, y, val_scores, val_y)
Output: results/phase4b/analysis/prf.csv

Usage: python scripts/prf_table.py
"""
import csv
import os

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_auc_score

ENCODERS = ["vits16_dino", "resnet50_dino", "vits16_moco", "resnet18_moco", "pixelstats", "resnet18_random"]
ATTACKERS = ["deep_mlp", "lightgbm", "svm_rbf"]
OUT = "results/phase4b/analysis/prf.csv"


def main():
    rows = []
    for enc in ENCODERS:
        best = None
        for att in ATTACKERS:
            f = f"results/preds/phase4b/seed0/{enc}/full/leakage_{att}.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f)
            val_auc = roc_auc_score(d["val_y"], d["val_scores"])  # selection on validation only
            if best is None or val_auc > best[0]:
                best = (val_auc, att, d)
        _, att, d = best
        # threshold that maximises F1 on validation
        p, r, thr = precision_recall_curve(d["val_y"], d["val_scores"])
        f1 = 2 * p * r / (p + r + 1e-12)
        t = thr[np.argmax(f1[:-1])]
        y, pred = d["y"], (d["scores"] >= t).astype(int)
        prec, rec, f, _ = precision_recall_fscore_support(y, pred, average="binary")
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows.append(dict(encoder=enc, attacker=att, threshold=round(float(t), 4),
                         auc=round(roc_auc_score(y, d["scores"]), 4), precision=round(prec, 4), recall=round(rec, 4),
                         f1=round(f, 4), accuracy=round(float((pred == y).mean()), 4),
                         tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
        print(rows[-1])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("written", OUT)


if __name__ == "__main__":
    main()
