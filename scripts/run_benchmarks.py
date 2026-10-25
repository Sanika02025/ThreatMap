"""
Executes full suite of controlled ablation experiments (Experiments A - H).
Usage:
    python scripts/run_benchmarks.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from threatmap.classification.pipeline import ThreatMapPipeline
from threatmap.config import load_config
from threatmap.data.tram_loader import load_benchmark_dataset
from threatmap.evaluation.benchmarks import run_ablation_benchmarks
from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.run_benchmarks")


def main():
    config = load_config(base_dir=BASE_DIR)
    pipeline = ThreatMapPipeline(config, base_dir=BASE_DIR)

    benchmark_path = BASE_DIR / "data" / "evaluation" / "tram_sample_benchmark.json"
    examples = load_benchmark_dataset(
        dataset_path=benchmark_path,
        valid_technique_ids=pipeline.kb.get_technique_ids(),
    )

    output_dir = BASE_DIR / "artifacts"
    logger.info("Starting ThreatMap Ablation Benchmarking Suite...")
    df = run_ablation_benchmarks(
        pipeline=pipeline,
        benchmark_examples=examples,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("THREATMAP ABLATION BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
