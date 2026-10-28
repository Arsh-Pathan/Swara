"""Evaluation and benchmarking subsystem."""

from swara.evaluation.metrics import compute_eer, compute_min_dcf
from swara.evaluation.harness import EvaluationHarness

__all__ = ["compute_eer", "compute_min_dcf", "EvaluationHarness"]
