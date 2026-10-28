"""Speech biometric evaluation metrics: EER, minDCF, and DER."""

from typing import List, Tuple
import numpy as np


def compute_eer(
    target_scores: List[float], nontarget_scores: List[float]
) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) and the corresponding threshold.

    Args:
        target_scores: List of cosine similarities between same-speaker pairs (positive trials).
        nontarget_scores: List of cosine similarities between different-speaker pairs (impostor trials).

    Returns:
        Tuple of (eer_percent, optimal_threshold).
    """
    targets = np.sort(np.array(target_scores, dtype=np.float32))
    nontargets = np.sort(np.array(nontarget_scores, dtype=np.float32))

    if len(targets) == 0 or len(nontargets) == 0:
        return 0.0, 0.68

    # Threshold sweep across range
    thresholds = np.linspace(-1.0, 1.0, 2001)
    frr = np.zeros_like(thresholds)
    far = np.zeros_like(thresholds)

    for i, t in enumerate(thresholds):
        # False Rejection Rate: fraction of targets with score < t
        frr[i] = np.mean(targets < t)
        # False Acceptance Rate: fraction of nontargets with score >= t
        far[i] = np.mean(nontargets >= t)

    diff = np.abs(frr - far)
    min_idx = int(np.argmin(diff))
    eer = float((frr[min_idx] + far[min_idx]) / 2.0 * 100.0)
    opt_thresh = float(thresholds[min_idx])

    return eer, opt_thresh


def compute_min_dcf(
    target_scores: List[float],
    nontarget_scores: List[float],
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Computes normalized Minimum Detection Cost Function (minDCF)."""
    targets = np.array(target_scores, dtype=np.float32)
    nontargets = np.array(nontarget_scores, dtype=np.float32)

    if len(targets) == 0 or len(nontargets) == 0:
        return 0.0

    thresholds = np.linspace(-1.0, 1.0, 1001)
    dcfs = []

    for t in thresholds:
        p_miss = np.mean(targets < t)
        p_fa = np.mean(nontargets >= t)
        dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1.0 - p_target)
        dcfs.append(dcf)

    dcf_default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf = min(dcfs) / dcf_default
    return float(min_dcf)
