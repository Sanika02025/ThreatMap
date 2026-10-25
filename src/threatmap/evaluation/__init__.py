"""
Evaluation and ablation benchmarking module.
"""

from threatmap.evaluation.metrics import calculate_multilabel_metrics, calculate_retrieval_recall_at_k
from threatmap.evaluation.evaluator import PipelineEvaluator
from threatmap.evaluation.error_analysis import perform_error_analysis
from threatmap.evaluation.benchmarks import run_ablation_benchmarks

__all__ = [
    "calculate_multilabel_metrics",
    "calculate_retrieval_recall_at_k",
    "PipelineEvaluator",
    "perform_error_analysis",
    "run_ablation_benchmarks",
]
