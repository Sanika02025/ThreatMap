"""
Loader and preprocessor for TRAM v1.3 multi-label dataset (19,178 sentences, 151 reports).
Reconstructs context from contiguous doc_title rows, handles duplicates and outliers,
and provides deterministic report-level sampling.
"""

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from threatmap.logging_config import setup_logger
from threatmap.schemas import BenchmarkExample, ProcessedSentence

logger = setup_logger("threatmap.tram_loader")


class TRAMRow(BaseModel):
    """Rich representation of a TRAM benchmark row with context."""
    global_index: int
    doc_title: str
    doc_sentence_index: int
    sentence: str
    labels: List[str] = Field(default_factory=list)
    is_labeled: bool = False
    is_duplicate: bool = False
    is_outlier: bool = False
    # Neighboring sentences within the same document
    prev_1: Optional[str] = None
    next_1: Optional[str] = None
    prev_2: Optional[str] = None
    next_2: Optional[str] = None

    def get_context_text(self, window_size: int = 1) -> str:
        """Returns neighboring context string according to paper specifications."""
        parts = []
        if window_size >= 2 and self.prev_2:
            parts.append(self.prev_2)
        if window_size >= 1 and self.prev_1:
            parts.append(self.prev_1)

        parts.append(self.sentence)

        if window_size >= 1 and self.next_1:
            parts.append(self.next_1)
        if window_size >= 2 and self.next_2:
            parts.append(self.next_2)

        return " ".join(parts)

    def to_processed_sentence(self) -> ProcessedSentence:
        """Converts to ThreatMap ProcessedSentence schema."""
        return ProcessedSentence(
            document_id=self.doc_title,
            sentence_id=f"S{self.global_index:05d}",
            sentence_index=self.doc_sentence_index,
            original_sentence=self.sentence,
            normalized_sentence=self.sentence.lower(),
            previous_sentence=self.prev_1,
            next_sentence=self.next_1,
            source_filename=self.doc_title,
        )

    def to_benchmark_example(self) -> BenchmarkExample:
        """Converts to ThreatMap BenchmarkExample schema."""
        return BenchmarkExample(
            id=f"TRAM-{self.global_index:05d}",
            text=self.sentence,
            labels=self.labels,
            split="test",
            source_report=self.doc_title,
        )


class TRAMDataset:
    """Full TRAM v1.3 dataset manager."""

    def __init__(self, rows: List[TRAMRow], metadata: Dict[str, Any]):
        self.rows = rows
        self.metadata = metadata
        self._doc_map: Dict[str, List[TRAMRow]] = {}
        for r in rows:
            self._doc_map.setdefault(r.doc_title, []).append(r)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TRAMRow:
        return self.rows[idx]

    @property
    def reports(self) -> List[str]:
        """Ordered list of unique document titles."""
        return list(self._doc_map.keys())

    def get_report_rows(self, doc_title: str) -> List[TRAMRow]:
        """Returns contiguous rows belonging to a specific threat report."""
        return self._doc_map.get(doc_title, [])

    def get_labeled_rows(self) -> List[TRAMRow]:
        """Returns only rows with at least one technique label."""
        return [r for r in self.rows if r.is_labeled]

    def get_unlabeled_rows(self) -> List[TRAMRow]:
        """Returns rows with no technique labels."""
        return [r for r in self.rows if not r.is_labeled]

    def sample_reports(self, n_reports: int = 30, seed: int = 42) -> "TRAMDataset":
        """
        Deterministically samples n whole reports to preserve document continuity.
        Recommended fallback when local hardware cannot process all 19k calls.
        """
        rng = random.Random(seed)
        all_docs = list(self.reports)
        sampled_docs = set(rng.sample(all_docs, min(n_reports, len(all_docs))))

        sampled_rows = [r for r in self.rows if r.doc_title in sampled_docs]
        meta = {
            **self.metadata,
            "sampled": True,
            "sampled_reports_count": len(sampled_docs),
            "sampled_sentences_count": len(sampled_rows),
            "seed": seed,
        }
        logger.info(
            f"Sampled {len(sampled_docs)} reports ({len(sampled_rows)} sentences) using seed={seed}"
        )
        return TRAMDataset(sampled_rows, meta)

    def to_benchmark_examples(self) -> List[BenchmarkExample]:
        return [r.to_benchmark_example() for r in self.rows]


def load_tram_dataset(
    file_path: str | Path,
    max_char_threshold: int = 10000,
) -> TRAMDataset:
    """
    Loads TRAM v1.3 multi_label.json with context reconstruction and invariant validation.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TRAM multi_label.json not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    logger.info(f"Loaded {len(raw_items)} raw items from {path.name}")

    # Count sentence occurrences to identify duplicate sentences
    sentence_counts = Counter(item.get("sentence", "") for item in raw_items)

    rows: List[TRAMRow] = []
    current_doc = None
    doc_sentence_idx = 0

    for idx, item in enumerate(raw_items):
        sentence = item.get("sentence", "")
        doc_title = item.get("doc_title", "Unknown_Report")
        labels = [str(l).strip() for l in item.get("labels", [])]

        if doc_title != current_doc:
            current_doc = doc_title
            doc_sentence_idx = 0
        else:
            doc_sentence_idx += 1

        is_dup = sentence_counts[sentence] > 1
        is_outlier = len(sentence) > max_char_threshold

        row = TRAMRow(
            global_index=idx,
            doc_title=doc_title,
            doc_sentence_index=doc_sentence_idx,
            sentence=sentence,
            labels=labels,
            is_labeled=len(labels) > 0,
            is_duplicate=is_dup,
            is_outlier=is_outlier,
        )
        rows.append(row)

    # Second pass: attach neighboring sentences within the same document
    for idx, row in enumerate(rows):
        # Previous 1
        if idx > 0 and rows[idx - 1].doc_title == row.doc_title:
            row.prev_1 = rows[idx - 1].sentence
            # Previous 2
            if idx > 1 and rows[idx - 2].doc_title == row.doc_title:
                row.prev_2 = rows[idx - 2].sentence

        # Next 1
        if idx + 1 < len(rows) and rows[idx + 1].doc_title == row.doc_title:
            row.next_1 = rows[idx + 1].sentence
            # Next 2
            if idx + 2 < len(rows) and rows[idx + 2].doc_title == row.doc_title:
                row.next_2 = rows[idx + 2].sentence

    # Metadata summary
    dup_count = sum(1 for r in rows if r.is_duplicate)
    outlier_count = sum(1 for r in rows if r.is_outlier)
    labeled_count = sum(1 for r in rows if r.is_labeled)
    all_label_instances = sum(len(r.labels) for r in rows)
    unique_classes = set(lbl for r in rows for lbl in r.labels)

    metadata = {
        "source_file": str(path),
        "total_sentences": len(rows),
        "labeled_sentences": labeled_count,
        "unlabeled_sentences": len(rows) - labeled_count,
        "total_label_instances": all_label_instances,
        "unique_classes_count": len(unique_classes),
        "unique_reports_count": len(set(r.doc_title for r in rows)),
        "duplicate_sentences_detected": dup_count,
        "outlier_sentences_detected": outlier_count,
    }

    logger.info(
        f"Initialized TRAMDataset: {len(rows)} sentences, {len(metadata['unique_reports_count'] if isinstance(metadata['unique_reports_count'], set) else [1])} reports, "
        f"{labeled_count} labeled ({all_label_instances} labels), {len(unique_classes)} classes."
    )

    return TRAMDataset(rows, metadata)


def load_benchmark_dataset(
    dataset_path: str | Path,
    valid_technique_ids: Optional[Set[str]] = None,
    split: Optional[str] = None,
) -> List[BenchmarkExample]:
    """
    Backwards-compatible loader function.
    Automatically routes TRAM multi_label.json to load_tram_dataset.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path}")

    # If it is the full TRAM dataset
    if "multi_label" in path.name:
        dataset = load_tram_dataset(path)
        examples = dataset.to_benchmark_examples()
    else:
        # Standard legacy JSON / JSONL loader
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            if isinstance(raw_data, dict):
                raw_data = raw_data.get("examples", [])
            examples = [BenchmarkExample(**item) for item in raw_data]

    if valid_technique_ids:
        validated = []
        for ex in examples:
            valid_labels = [l for l in ex.labels if l in valid_technique_ids]
            ex.labels = valid_labels
            validated.append(ex)
        examples = validated

    return examples
