"""
Evaluator for measuring pipeline performance on labeled datasets.
"""

import time
from typing import Dict, List, Set, Tuple
from threatmap.classification.pipeline import ThreatMapPipeline
from threatmap.logging_config import setup_logger
from threatmap.schemas import BenchmarkExample, EvaluationMetrics, ProcessedSentence, SentenceAnalysisResult
from threatmap.evaluation.error_analysis import perform_error_analysis
from threatmap.evaluation.metrics import (
    calculate_mean_reciprocal_rank,
    calculate_multilabel_metrics,
    calculate_retrieval_recall_at_k,
    safe_divide,
)

logger = setup_logger("threatmap.evaluator")


class PipelineEvaluator:
    """Runs evaluation benchmark suites across the ThreatMap pipeline."""

    def __init__(self, pipeline: ThreatMapPipeline):
        self.pipeline = pipeline

    def evaluate(
        self,
        examples: List[BenchmarkExample],
        retrieval_mode: str = "hybrid_rrf",
        top_k: int = 5,
        use_llm: bool = False,
        context_strategy: str = "target_with_context_and_candidates",
    ) -> Tuple[EvaluationMetrics, List[SentenceAnalysisResult], Dict]:
        """
        Executes pipeline over benchmark examples and computes standard metrics.
        """
        logger.info(
            f"Evaluating {len(examples)} examples (mode={retrieval_mode}, top_k={top_k}, use_llm={use_llm})..."
        )

        y_true: List[Set[str]] = [set(ex.labels) for ex in examples]
        y_pred: List[Set[str]] = []
        retrieved_candidates_list = []
        sentence_results: List[SentenceAnalysisResult] = []

        start_time = time.time()

        for idx, ex in enumerate(examples):
            # Create ProcessedSentence representation
            sent_obj = ProcessedSentence(
                document_id="EVAL",
                sentence_id=f"E{idx + 1:03d}",
                sentence_index=idx,
                original_sentence=ex.text,
                normalized_sentence=ex.text.lower(),
                previous_sentence=None,
                next_sentence=None,
                source_filename=ex.source_report,
            )

            res = self.pipeline.analyze_sentence(
                sentence=sent_obj,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                use_llm=use_llm,
                context_strategy=context_strategy,
            )
            sentence_results.append(res)
            retrieved_candidates_list.append(res.alternative_candidates)

            preds = {p.technique_id for p in res.predictions} if not res.abstained else set()
            y_pred.append(preds)

        elapsed_seconds = time.time() - start_time

        # Multilabel classification metrics
        micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, per_tech = (
            calculate_multilabel_metrics(y_true, y_pred, zero_division=0.0)
        )

        # Retrieval Recall@k
        rec_1 = calculate_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=1)
        rec_3 = calculate_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=3)
        rec_5 = calculate_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=5)
        rec_10 = calculate_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=10)
        mrr = calculate_mean_reciprocal_rank(y_true, retrieved_candidates_list)

        # Coverage and abstention
        non_abstained_count = sum(1 for res in sentence_results if not res.abstained)
        coverage = safe_divide(non_abstained_count, len(examples))
        abstention_rate = safe_divide(len(examples) - non_abstained_count, len(examples))

        # Qualitative error analysis
        error_details = perform_error_analysis(examples, sentence_results)

        metrics = EvaluationMetrics(
            micro_precision=micro_p,
            micro_recall=micro_r,
            micro_f1=micro_f1,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            recall_at_1=round(rec_1, 4),
            recall_at_3=round(rec_3, 4),
            recall_at_5=round(rec_5, 4),
            recall_at_10=round(rec_10, 4),
            mrr=mrr,
            coverage=round(coverage, 4),
            abstention_rate=round(abstention_rate, 4),
            per_technique=per_tech,
            total_examples=len(examples),
            evaluated_techniques_count=len(per_tech),
            runtime_seconds=round(elapsed_seconds, 3),
        )

        return metrics, sentence_results, error_details
