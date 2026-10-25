"""
Multilabel evaluation metrics and retrieval ranking metrics for ThreatMap.
Strictly handles zero-division edge cases and maintains reproducible conventions.
"""

from typing import Dict, List, Set, Tuple
from threatmap.schemas import EvaluationMetrics, RetrievedCandidate


def safe_divide(numerator: float, denominator: float, zero_val: float = 0.0) -> float:
    """Safe division preventing ZeroDivisionError."""
    if denominator == 0 or denominator == 0.0:
        return zero_val
    return numerator / denominator


def calculate_multilabel_metrics(
    y_true: List[Set[str]],
    y_pred: List[Set[str]],
    zero_division: float = 0.0,
) -> Tuple[float, float, float, float, float, float, Dict[str, Dict[str, float]]]:
    """
    Computes micro and macro Precision, Recall, and F1 scores across multilabel predictions.
    Returns: (micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, per_technique_dict)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # Collect all unique technique classes appearing in either truth or predictions
    all_classes: Set[str] = set()
    for labels in y_true:
        all_classes.update(labels)
    for labels in y_pred:
        all_classes.update(labels)

    if not all_classes:
        # Both sets were completely empty of labels
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_technique: Dict[str, Dict[str, float]] = {}
    precisions = []
    recalls = []
    f1s = []

    for cls in sorted(list(all_classes)):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if cls in yt and cls in yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if cls not in yt and cls in yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if cls in yt and cls not in yp)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = safe_divide(tp, tp + fp, zero_val=zero_division)
        r = safe_divide(tp, tp + fn, zero_val=zero_division)
        f1 = safe_divide(2 * p * r, p + r, zero_val=zero_division)

        per_technique[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": sum(1 for yt in y_true if cls in yt),
        }

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    # Micro averages
    micro_p = safe_divide(total_tp, total_tp + total_fp, zero_val=zero_division)
    micro_r = safe_divide(total_tp, total_tp + total_fn, zero_val=zero_division)
    micro_f1 = safe_divide(2 * micro_p * micro_r, micro_p + micro_r, zero_val=zero_division)

    # Macro averages (unweighted average across classes)
    macro_p = sum(precisions) / len(precisions) if precisions else 0.0
    macro_r = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return (
        round(micro_p, 4),
        round(micro_r, 4),
        round(micro_f1, 4),
        round(macro_p, 4),
        round(macro_r, 4),
        round(macro_f1, 4),
        per_technique,
    )


def calculate_retrieval_recall_at_k(
    y_true: List[Set[str]],
    retrieved_candidates_list: List[List[RetrievedCandidate]],
    k: int = 5,
) -> float:
    """
    Computes Retrieval Recall@k:
    Fraction of labeled examples where at least one ground truth technique ID appears in the top-k retrieved list.
    """
    total_labeled = 0
    hits = 0

    for labels, candidates in zip(y_true, retrieved_candidates_list):
        if not labels:
            continue  # Negative/abstention examples do not count towards retrieval recall of techniques
        total_labeled += 1
        top_k_ids = {c.technique_id for c in candidates[:k]}
        if top_k_ids.intersection(labels):
            hits += 1

    return safe_divide(hits, total_labeled, zero_val=0.0)


def calculate_mean_reciprocal_rank(
    y_true: List[Set[str]],
    retrieved_candidates_list: List[List[RetrievedCandidate]],
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for the first relevant technique found.
    MRR = (1 / |Q|) * sum(1 / rank_i)
    """
    reciprocal_ranks = []

    for labels, candidates in zip(y_true, retrieved_candidates_list):
        if not labels:
            continue

        found_rank = None
        for rank, cand in enumerate(candidates, start=1):
            if cand.technique_id in labels:
                found_rank = rank
                break

        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
