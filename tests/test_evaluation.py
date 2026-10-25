"""
Unit tests for multilabel evaluation metrics, Retrieval Recall@k, and MRR.
"""

from threatmap.evaluation.metrics import (
    calculate_mean_reciprocal_rank,
    calculate_multilabel_metrics,
    calculate_retrieval_recall_at_k,
    safe_divide,
)
from threatmap.schemas import RetrievedCandidate


def test_safe_divide():
    assert safe_divide(10.0, 2.0) == 5.0
    assert safe_divide(1.0, 0.0, zero_val=0.0) == 0.0


def test_calculate_multilabel_metrics_exact_match():
    y_true = [{"T1059.001"}, {"T1003.001", "T1033"}, set()]
    y_pred = [{"T1059.001"}, {"T1003.001", "T1033"}, set()]

    micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, per_tech = (
        calculate_multilabel_metrics(y_true, y_pred)
    )
    assert micro_p == 1.0
    assert micro_r == 1.0
    assert micro_f1 == 1.0
    assert macro_f1 == 1.0
    assert "T1059.001" in per_tech
    assert per_tech["T1059.001"]["f1"] == 1.0


def test_calculate_multilabel_metrics_partial_match():
    y_true = [{"T1059.001"}, {"T1003.001"}]
    y_pred = [{"T1059.001"}, {"T1053.005"}]  # Second example is FP + FN

    micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, _ = (
        calculate_multilabel_metrics(y_true, y_pred)
    )
    # TP=1, FP=1, FN=1 -> Micro P=0.5, Micro R=0.5, Micro F1=0.5
    assert micro_p == 0.5
    assert micro_r == 0.5
    assert micro_f1 == 0.5


def test_calculate_retrieval_recall_at_k_and_mrr():
    y_true = [{"T1059.001"}, {"T1003.001"}]

    def make_cand(tid, rank):
        return RetrievedCandidate(
            technique_id=tid,
            technique_name=tid,
            tactics=[],
            description="",
            fused_rank=rank,
            fused_score=1.0 / rank,
        )

    # First query has target at rank 1; second query has target at rank 3
    retrieved = [
        [make_cand("T1059.001", 1), make_cand("T1053.005", 2)],
        [make_cand("T1033", 1), make_cand("T1021.001", 2), make_cand("T1003.001", 3)],
    ]

    # Recall@1: query 1 has target at 1 (hit), query 2 does not (miss) -> 1/2 = 0.5
    rec_1 = calculate_retrieval_recall_at_k(y_true, retrieved, k=1)
    assert rec_1 == 0.5

    # Recall@3: both hit in top 3 -> 2/2 = 1.0
    rec_3 = calculate_retrieval_recall_at_k(y_true, retrieved, k=3)
    assert rec_3 == 1.0

    # MRR: (1/1 + 1/3) / 2 = (1 + 0.3333)/2 = 0.6667
    mrr = calculate_mean_reciprocal_rank(y_true, retrieved)
    assert mrr == 0.6667
