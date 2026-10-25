"""
Loader for TRAM and custom threat intelligence benchmark datasets.
"""

import json
from pathlib import Path
from typing import List, Optional, Set
from threatmap.logging_config import setup_logger
from threatmap.schemas import BenchmarkExample
from threatmap.data.validators import validate_technique_id

logger = setup_logger("threatmap.tram_loader")


def load_benchmark_dataset(
    dataset_path: str | Path,
    valid_technique_ids: Optional[Set[str]] = None,
    split: Optional[str] = None,
) -> List[BenchmarkExample]:
    """
    Loads benchmark examples from a JSON or JSONL file.
    Optionally filters by split and validates labels against loaded ATT&CK KB.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path}")

    examples: List[BenchmarkExample] = []

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                data = json.loads(clean_line)
                examples.append(BenchmarkExample(**data))
        else:
            raw_data = json.load(f)
            if isinstance(raw_data, dict):
                raw_data = raw_data.get("examples", [])
            for item in raw_data:
                examples.append(BenchmarkExample(**item))

    # Optional split filtering
    if split:
        examples = [ex for ex in examples if ex.split == split]

    # Validate labels if KB technique IDs provided
    if valid_technique_ids:
        validated_examples = []
        for ex in examples:
            valid_labels = [lbl for lbl in ex.labels if validate_technique_id(lbl, valid_technique_ids)]
            invalid_labels = set(ex.labels) - set(valid_labels)
            if invalid_labels:
                logger.warning(
                    f"Example {ex.id} contains labels not present in knowledge base: {invalid_labels}. Retaining valid labels: {valid_labels}"
                )
            ex.labels = valid_labels
            validated_examples.append(ex)
        examples = validated_examples

    logger.info(f"Loaded {len(examples)} benchmark examples from {path.name}")
    return examples
