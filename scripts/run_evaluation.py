"""
Runs pipeline evaluation on labeled benchmark dataset and displays quantitative metrics.
Usage:
    python scripts/run_evaluation.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from threatmap.classification.pipeline import ThreatMapPipeline
from threatmap.config import load_config
from threatmap.data.tram_loader import load_benchmark_dataset
from threatmap.evaluation.evaluator import PipelineEvaluator
from threatmap.evaluation.report_generator import generate_evaluation_markdown_report
from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.run_evaluation")


def main():
    config = load_config(base_dir=BASE_DIR)
    pipeline = ThreatMapPipeline(config, base_dir=BASE_DIR)

    benchmark_path = BASE_DIR / "data" / "evaluation" / "tram_sample_benchmark.json"
    if not benchmark_path.exists():
        logger.error(f"Benchmark dataset not found at {benchmark_path}")
        sys.exit(1)

    examples = load_benchmark_dataset(
        dataset_path=benchmark_path,
        valid_technique_ids=pipeline.kb.get_technique_ids(),
    )

    evaluator = PipelineEvaluator(pipeline)
    metrics, results, error_details = evaluator.evaluate(
        examples=examples,
        retrieval_mode=config.retrieval.hybrid_mode,
        top_k=config.retrieval.top_k,
        use_llm=False,  # Fallback retrieval-only evaluation
    )

    metadata = {
        "dataset_name": "TRAM Sample Benchmark",
        "retrieval_mode": config.retrieval.hybrid_mode,
        "top_k": config.retrieval.top_k,
        "use_llm": False,
    }

    report_path = BASE_DIR / "artifacts" / "evaluation_report.md"
    report_text = generate_evaluation_markdown_report(
        metrics=metrics,
        error_analysis=error_details,
        metadata=metadata,
        file_path=report_path,
    )

    print("\n" + "=" * 60)
    print("THREATMAP BENCHMARK EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Examples Evaluated: {metrics.total_examples}")
    print(f"Micro F1:                {metrics.micro_f1:.4f}")
    print(f"Macro F1:                {metrics.macro_f1:.4f}")
    print(f"Retrieval Recall@1:      {metrics.recall_at_1:.4f}")
    print(f"Retrieval Recall@5:      {metrics.recall_at_5:.4f}")
    print(f"Mean Reciprocal Rank:    {metrics.mrr:.4f}")
    print(f"Coverage:                {metrics.coverage * 100:.1f}%")
    print(f"Abstention Rate:         {metrics.abstention_rate * 100:.1f}%")
    print(f"Total Runtime:           {metrics.runtime_seconds:.3f} s")
    print("=" * 60)
    print(f"Report written to: {report_path}\n")


if __name__ == "__main__":
    main()
