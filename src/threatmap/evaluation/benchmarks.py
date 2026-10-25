"""
Ablation experiment runner for comparative evaluation (Experiments A - H).
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from threatmap.classification.pipeline import ThreatMapPipeline
from threatmap.logging_config import setup_logger
from threatmap.schemas import BenchmarkExample
from threatmap.evaluation.evaluator import PipelineEvaluator

logger = setup_logger("threatmap.benchmarks")


def run_ablation_benchmarks(
    pipeline: ThreatMapPipeline,
    benchmark_examples: List[BenchmarkExample],
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Executes controlled ablation experiments across retrieval and classification configurations.
    """
    evaluator = PipelineEvaluator(pipeline)
    results_summary: List[Dict[str, Any]] = []

    # Experiment Definitions
    experiments = [
        {
            "id": "Exp A",
            "name": "Lexical Retrieval Only (BM25)",
            "retrieval_mode": "lexical",
            "top_k": 5,
            "use_llm": False,
            "reranker": False,
            "context_strategy": "target_only",
        },
        {
            "id": "Exp B",
            "name": "Semantic Retrieval Only (Dense)",
            "retrieval_mode": "semantic",
            "top_k": 5,
            "use_llm": False,
            "reranker": False,
            "context_strategy": "target_only",
        },
        {
            "id": "Exp C",
            "name": "Hybrid Retrieval (RRF)",
            "retrieval_mode": "hybrid_rrf",
            "top_k": 5,
            "use_llm": False,
            "reranker": False,
            "context_strategy": "target_only",
        },
        {
            "id": "Exp D",
            "name": "Hybrid (Weighted Score Fusion)",
            "retrieval_mode": "hybrid_weighted",
            "top_k": 5,
            "use_llm": False,
            "reranker": False,
            "context_strategy": "target_only",
        },
        {
            "id": "Exp E",
            "name": "Hybrid + Cross-Encoder Reranker",
            "retrieval_mode": "hybrid_rrf",
            "top_k": 5,
            "use_llm": False,
            "reranker": True,
            "context_strategy": "target_only",
        },
        {
            "id": "Exp F",
            "name": "Hybrid + Local LLM Classification",
            "retrieval_mode": "hybrid_rrf",
            "top_k": 5,
            "use_llm": True,
            "reranker": False,
            "context_strategy": "target_with_candidates",
        },
        {
            "id": "Exp G",
            "name": "Hybrid + Context-Aware LLM Classification",
            "retrieval_mode": "hybrid_rrf",
            "top_k": 5,
            "use_llm": True,
            "reranker": False,
            "context_strategy": "target_with_context_and_candidates",
        },
        {
            "id": "Exp H",
            "name": "Hybrid + Calibrated Abstention",
            "retrieval_mode": "hybrid_rrf",
            "top_k": 3,
            "use_llm": False,
            "reranker": False,
            "context_strategy": "target_only",
        },
    ]

    llm_available = pipeline.llm_client.is_available()

    for exp in experiments:
        logger.info(f"Running Ablation {exp['id']}: {exp['name']}...")

        # If experiment requires LLM but Ollama is not available, mark clearly
        if exp["use_llm"] and not llm_available:
            results_summary.append(
                {
                    "Experiment ID": exp["id"],
                    "Method": exp["name"],
                    "Micro F1": None,
                    "Macro F1": None,
                    "Recall@1": None,
                    "Recall@5": None,
                    "MRR": None,
                    "Coverage": None,
                    "Abstention Rate": None,
                    "Runtime (s)": 0.0,
                    "Status": "Skipped (Ollama offline or uninstalled)",
                }
            )
            continue

        # Configure reranker state temporarily
        original_reranker_state = pipeline.reranker.enabled
        pipeline.reranker.enabled = exp["reranker"]

        try:
            metrics, _, _ = evaluator.evaluate(
                examples=benchmark_examples,
                retrieval_mode=exp["retrieval_mode"],
                top_k=exp["top_k"],
                use_llm=exp["use_llm"],
                context_strategy=exp["context_strategy"],
            )

            results_summary.append(
                {
                    "Experiment ID": exp["id"],
                    "Method": exp["name"],
                    "Micro F1": metrics.micro_f1,
                    "Macro F1": metrics.macro_f1,
                    "Recall@1": metrics.recall_at_1,
                    "Recall@5": metrics.recall_at_5,
                    "MRR": metrics.mrr,
                    "Coverage": metrics.coverage,
                    "Abstention Rate": metrics.abstention_rate,
                    "Runtime (s)": metrics.runtime_seconds,
                    "Status": "Completed",
                }
            )
        except Exception as e:
            logger.error(f"Experiment {exp['id']} failed: {e}")
            results_summary.append(
                {
                    "Experiment ID": exp["id"],
                    "Method": exp["name"],
                    "Micro F1": None,
                    "Macro F1": None,
                    "Recall@1": None,
                    "Recall@5": None,
                    "MRR": None,
                    "Coverage": None,
                    "Abstention Rate": None,
                    "Runtime (s)": 0.0,
                    "Status": f"Failed: {str(e)}",
                }
            )
        finally:
            pipeline.reranker.enabled = original_reranker_state

    df = pd.DataFrame(results_summary)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_file = out_path / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved ablation benchmark results to {csv_file}")

    return df
