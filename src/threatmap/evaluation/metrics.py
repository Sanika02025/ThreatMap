"""
Multilabel evaluation metrics and retrieval ranking metrics for ThreatMap.
Conforms strictly to RAG-ATT&CK (Schuurman, 2025) reproduction protocol:
1. Retrieval Recall@k: Fraction of true technique labels retrieved in top-k.
2. Precision@1 and Precision@k.
3. Macro & Micro metrics restricted to the 50 technique classes, with unlisted predictions audited separately.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from threatmap.schemas import RetrievedCandidate


def safe_divide(numerator: float, denominator: float, zero_val: float = 0.0) -> float:
    """Safe division preventing ZeroDivisionError."""
    if denominator == 0 or denominator == 0.0:
        return zero_val
    return numerator / denominator


def calculate_paper_retrieval_recall_at_k(
    y_true: List[Set[str]],
    retrieved_candidates_list: List[List[Union[RetrievedCandidate, str]]],
    k: int = 10,
) -> float:
    """
    RAG-ATT&CK Paper Recall@k Formulation:
    Counts the total fraction of true labels found within the top-k retrieved techniques.
    Denominator = total ground truth label instances (e.g. 5,143 in TRAM).
    """
    total_true_labels = sum(len(yt) for yt in y_true)
    if total_true_labels == 0:
        return 0.0

    hits = 0
    for true_labels, candidates in zip(y_true, retrieved_candidates_list):
        if not true_labels:
            continue

        # Extract technique IDs from candidate objects or strings
        top_k_cands = candidates[:k]
        top_k_ids = set()
        for c in top_k_cands:
            if isinstance(c, RetrievedCandidate):
                top_k_ids.add(c.technique_id)
            elif isinstance(c, str):
                top_k_ids.add(c)
            elif isinstance(c, dict):
                top_k_ids.add(c.get("technique_id", ""))

        # Count how many of the true labels are in top-k
        true_set = set(true_labels) if not isinstance(true_labels, set) else true_labels
        overlap = true_set.intersection(top_k_ids)
        hits += len(overlap)

    return safe_divide(hits, total_true_labels, zero_val=0.0)


def calculate_precision_at_k(
    y_true: List[Set[str]],
    retrieved_candidates_list: List[List[Union[RetrievedCandidate, str]]],
    k: int = 1,
) -> float:
    """
    Precision@k: Fraction of top-k retrieved candidates that are true labels.
    """
    total_retrieved = 0
    true_retrieved = 0

    for true_labels, candidates in zip(y_true, retrieved_candidates_list):
        top_k_cands = candidates[:k]
        for c in top_k_cands:
            cid = c.technique_id if isinstance(c, RetrievedCandidate) else (c.get("technique_id") if isinstance(c, dict) else str(c))
            total_retrieved += 1
            if cid in true_labels:
                true_retrieved += 1

    return safe_divide(true_retrieved, total_retrieved, zero_val=0.0)


def calculate_retrieval_recall_at_k(
    y_true: List[Set[str]],
    retrieved_candidates_list: List[List[Union[RetrievedCandidate, str]]],
    k: int = 5,
    method: str = "paper_fraction",  # 'paper_fraction' (paper) or 'any_hit' (legacy)
) -> float:
    """
    Retrieval Recall@k supporting both paper protocol and legacy hit-rate mode.
    """
    if method == "paper_fraction":
        return calculate_paper_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=k)

    # Legacy: fraction of labeled sentences where at least one label was retrieved
    total_labeled = 0
    hits = 0

    for labels, candidates in zip(y_true, retrieved_candidates_list):
        if not labels:
            continue
        total_labeled += 1
        top_k_ids = set()
        for c in candidates[:k]:
            cid = c.technique_id if isinstance(c, RetrievedCandidate) else (c.get("technique_id") if isinstance(c, dict) else str(c))
            top_k_ids.add(cid)

        if top_k_ids.intersection(labels):
            hits += 1

    return safe_divide(hits, total_labeled, zero_val=0.0)


def calculate_mean_reciprocal_rank(
    y_true: List[Set[str]],
    retrieved_candidates_list: List[List[Union[RetrievedCandidate, str]]],
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for the first relevant technique found.
    """
    reciprocal_ranks = []

    for labels, candidates in zip(y_true, retrieved_candidates_list):
        if not labels:
            continue

        found_rank = None
        for rank, cand in enumerate(candidates, start=1):
            cid = cand.technique_id if isinstance(cand, RetrievedCandidate) else (cand.get("technique_id") if isinstance(cand, dict) else str(cand))
            if cid in labels:
                found_rank = rank
                break

        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def calculate_reproduction_metrics(
    y_true: List[Set[str]],
    y_pred: List[Set[str]],
    valid_50_classes: Set[str],
    raw_predictions: Optional[List[List[str]]] = None,
    zero_division: float = 0.0,
) -> Dict[str, Any]:
    """
    Strict RAG-ATT&CK Table 2 Evaluation Protocol:
    1. Metrics evaluated strictly over the 50 classes.
    2. Unlisted/out-of-scope predictions are audited separately.
    3. Handles 19,178 sentences (including unlabelled ones).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # Audit unlisted predictions if raw predictions provided
    total_raw_pred_count = 0
    unlisted_pred_count = 0
    preds_to_audit = raw_predictions if raw_predictions is not None else y_pred
    for r_list in preds_to_audit:
        for p in r_list:
            total_raw_pred_count += 1
            if p not in valid_50_classes:
                unlisted_pred_count += 1

    # Filter true and pred sets strictly to the valid 50 classes
    clean_y_true = [{lbl for lbl in yt if lbl in valid_50_classes} for yt in y_true]
    clean_y_pred = [{lbl for lbl in yp if lbl in valid_50_classes} for yp in y_pred]

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_technique: Dict[str, Dict[str, float]] = {}
    precisions = []
    recalls = []
    f1s = []

    for cls in sorted(list(valid_50_classes)):
        tp = sum(1 for yt, yp in zip(clean_y_true, clean_y_pred) if cls in yt and cls in yp)
        fp = sum(1 for yt, yp in zip(clean_y_true, clean_y_pred) if cls not in yt and cls in yp)
        fn = sum(1 for yt, yp in zip(clean_y_true, clean_y_pred) if cls in yt and cls not in yp)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = safe_divide(tp, tp + fp, zero_val=zero_division)
        r = safe_divide(tp, tp + fn, zero_val=zero_division)
        f1 = safe_divide(2 * p * r, p + r, zero_val=zero_division)

        support = sum(1 for yt in clean_y_true if cls in yt)
        per_technique[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    # Micro averages over the 50 classes
    micro_p = safe_divide(total_tp, total_tp + total_fp, zero_val=zero_division)
    micro_r = safe_divide(total_tp, total_tp + total_fn, zero_val=zero_division)
    micro_f1 = safe_divide(2 * micro_p * micro_r, micro_p + micro_r, zero_val=zero_division)

    # Macro averages (unweighted average across all 50 classes)
    macro_p = sum(precisions) / len(precisions) if precisions else 0.0
    macro_r = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "micro_precision": round(micro_p * 100, 2),
        "micro_recall": round(micro_r * 100, 2),
        "micro_f1": round(micro_f1 * 100, 2),
        "macro_precision": round(macro_p * 100, 2),
        "macro_recall": round(macro_r * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_true_instances": total_tp + total_fn,
        "total_raw_predictions": total_raw_pred_count,
        "unlisted_predictions": unlisted_pred_count,
        "per_technique": per_technique,
    }


def calculate_multilabel_metrics(
    y_true: List[Set[str]],
    y_pred: List[Set[str]],
    zero_division: float = 0.0,
) -> Tuple[float, float, float, float, float, float, Dict[str, Dict[str, float]]]:
    """
    Backwards-compatible wrapper returning tuple for existing pipeline/evaluator.
    """
    all_classes: Set[str] = set()
    for labels in y_true:
        all_classes.update(labels)
    for labels in y_pred:
        all_classes.update(labels)

    if not all_classes:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, {}

    res = calculate_reproduction_metrics(
        y_true=y_true,
        y_pred=y_pred,
        valid_50_classes=all_classes,
        zero_division=zero_division,
    )

    return (
        round(res["micro_precision"] / 100, 4),
        round(res["micro_recall"] / 100, 4),
        round(res["micro_f1"] / 100, 4),
        round(res["macro_precision"] / 100, 4),
        round(res["macro_recall"] / 100, 4),
        round(res["macro_f1"] / 100, 4),
        res["per_technique"],
    )
