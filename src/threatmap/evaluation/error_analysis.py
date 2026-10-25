"""
Error analysis engine for diagnosing false positives, false negatives, and retrieval misses.
"""

from typing import Any, Dict, List, Set
from threatmap.schemas import BenchmarkExample, SentenceAnalysisResult


def perform_error_analysis(
    examples: List[BenchmarkExample],
    results: List[SentenceAnalysisResult],
) -> Dict[str, Any]:
    """
    Performs granular qualitative and quantitative error analysis across predictions.
    """
    false_positives: List[Dict[str, Any]] = []
    false_negatives: List[Dict[str, Any]] = []
    abstention_breakdown: List[Dict[str, Any]] = []

    for ex, res in zip(examples, results):
        true_labels: Set[str] = set(ex.labels)
        pred_labels: Set[str] = {p.technique_id for p in res.predictions}

        # Track Abstentions
        if res.abstained:
            abstention_breakdown.append(
                {
                    "example_id": ex.id,
                    "text": ex.text,
                    "ground_truth": list(true_labels),
                    "abstention_reason": res.abstention_reason,
                    "was_correct_abstention": len(true_labels) == 0,
                }
            )

        # False Positives: predicted but not in truth
        fp_ids = pred_labels - true_labels
        for fp_id in fp_ids:
            pred_obj = next((p for p in res.predictions if p.technique_id == fp_id), None)
            false_positives.append(
                {
                    "example_id": ex.id,
                    "text": ex.text,
                    "predicted_technique_id": fp_id,
                    "predicted_technique_name": pred_obj.technique_name if pred_obj else "",
                    "ground_truth": list(true_labels),
                    "evidence_quote": pred_obj.evidence_quote if pred_obj else "",
                    "explanation": pred_obj.explanation if pred_obj else "",
                }
            )

        # False Negatives: in truth but missed in predictions
        fn_ids = true_labels - pred_labels
        for fn_id in fn_ids:
            # Check if it was in the retrieved candidates
            candidate_ranks = [
                idx + 1
                for idx, c in enumerate(res.alternative_candidates)
                if c.technique_id == fn_id
            ]
            retrieved_rank = candidate_ranks[0] if candidate_ranks else None
            false_negatives.append(
                {
                    "example_id": ex.id,
                    "text": ex.text,
                    "missed_technique_id": fn_id,
                    "retrieved_in_candidates": retrieved_rank is not None,
                    "retrieved_rank": retrieved_rank,
                    "abstained": res.abstained,
                    "abstention_reason": res.abstention_reason,
                }
            )

    return {
        "total_examples": len(examples),
        "total_false_positives": len(false_positives),
        "total_false_negatives": len(false_negatives),
        "total_abstentions": len(abstention_breakdown),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "abstention_details": abstention_breakdown,
    }
