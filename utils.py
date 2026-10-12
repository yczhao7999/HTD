"""Evaluation metrics for hyperspectral target detection."""

import numpy as np
from sklearn.metrics import auc, roc_curve


def _normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    minimum = np.nanmin(scores)
    maximum = np.nanmax(scores)
    return (scores - minimum) / (maximum - minimum + 1e-12)


def detection_metrics(labels, scores):

    labels = np.asarray(labels).reshape(-1).astype(bool)
    scores = np.asarray(scores).reshape(-1).astype(np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same number of elements")
    if labels.all() or (~labels).all():
        raise ValueError("labels must contain both target and background pixels")

    normalized_scores = _normalize_scores(scores)

    fpr, tpr, thresholds = roc_curve(labels, normalized_scores)
    full_fpr, full_tpr = fpr, tpr

    # SelfMTL convention: drop the first (inf-threshold) point, integrate the
    # ROC branches over the score axis, and use a unit offset for AUC-ODP.
    fpr = fpr[1:]
    tpr = tpr[1:]
    thresholds = thresholds[1:]

    roc_auc = float(auc(fpr, tpr))
    auc_f = float(auc(thresholds, fpr))
    auc_d = float(auc(thresholds, tpr))
    auc_td = float(auc_d + roc_auc)
    auc_bs = float(roc_auc - auc_f)
    auc_odp = float(1.0 + auc_d - auc_f)

    def tpr_at(maximum_fpr):
        return float(np.interp(maximum_fpr, full_fpr, full_tpr))

    def normalized_partial_auc(maximum_fpr):
        inside = full_fpr < maximum_fpr
        clipped_fpr = np.concatenate((full_fpr[inside], [maximum_fpr]))
        clipped_tpr = np.concatenate(
            (full_tpr[inside], [np.interp(maximum_fpr, full_fpr, full_tpr)])
        )
        return float(auc(clipped_fpr, clipped_tpr) / maximum_fpr)

    return {
        "roc_auc": roc_auc,
        "auc_f": auc_f,
        "auc_d": auc_d,
        "auc_td": auc_td,
        "auc_bs": auc_bs,
        "auc_odp": auc_odp,
        "pauc_fpr_1pct": normalized_partial_auc(0.01),
        "tpr_at_fpr_0_1pct": tpr_at(0.001),
        "tpr_at_fpr_1pct": tpr_at(0.01),
    }