from __future__ import annotations

import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def quantile_threshold(scores: np.ndarray, quantile: float = 0.995) -> float:
    return float(np.quantile(np.asarray(scores, dtype=float), quantile))


def evt_pot_threshold(
    scores: np.ndarray,
    tail_quantile: float = 0.98,
    target_tail_probability: float = 0.005,
) -> float:
    """Estimate a high normal-score threshold with EVT peak-over-threshold fitting."""
    scores = np.asarray(scores, dtype=float)
    fallback = quantile_threshold(scores, 1.0 - target_tail_probability)
    base = float(np.quantile(scores, tail_quantile))
    excess = scores[scores > base] - base
    tail_probability = len(excess) / len(scores)
    if len(excess) < 20 or tail_probability <= target_tail_probability:
        return fallback

    shape, _, scale = genpareto.fit(excess, floc=0)
    if not np.isfinite(scale) or scale <= 0:
        return fallback

    conditional_tail = target_tail_probability / tail_probability
    threshold = base + float(genpareto.isf(conditional_tail, shape, loc=0, scale=scale))
    if not np.isfinite(threshold):
        return fallback
    return threshold


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    adjusted = y_pred.astype(int).copy()
    y_true = y_true.astype(int)
    start = None
    for i, value in enumerate(y_true):
        if value == 1 and start is None:
            start = i
        if start is not None and (value == 0 or i == len(y_true) - 1):
            end = i if value == 0 else i + 1
            if adjusted[start:end].any():
                adjusted[start:end] = 1
            start = None
    return adjusted


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (scores >= threshold).astype(int)
    y_adj = point_adjust(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true, y_adj, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pa_precision": float(precision_score(y_true, y_adj, zero_division=0)),
        "pa_recall": float(recall_score(y_true, y_adj, zero_division=0)),
        "pa_f1": float(f1_score(y_true, y_adj, zero_division=0)),
        "pa_accuracy": float(accuracy_score(y_true, y_adj)),
        "pa_tn": int(tn_a),
        "pa_fp": int(fp_a),
        "pa_fn": int(fn_a),
        "pa_tp": int(tp_a),
    }


def roc_pr_curves(y_true: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}

