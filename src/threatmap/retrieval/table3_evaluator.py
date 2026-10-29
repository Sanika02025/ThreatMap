"""
Evaluation harness for reproducing Table 3 of RAG-ATT&CK (Schuurman, 2025).
Evaluates retrieval across:
- Embedding models: nomic-embed-text-v1.5, ATTACK-BERT, all-MiniLM-L6-v2
- Description formats: full, first_paragraph, summarized, chunks_500, chunks_1000
- Reranking: cross-encoder reranker over top 15
- Metrics: Recall@1, Recall@3, Recall@5, Recall@10, Precision@1, MRR
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from threatmap.data.tram_loader import TRAMDataset, load_tram_dataset
from threatmap.evaluation.metrics import (
    calculate_paper_retrieval_recall_at_k,
    calculate_precision_at_k,
    calculate_mean_reciprocal_rank,
    safe_divide,
)
from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.table3_evaluator")

# Official paper targets for Table 3
TABLE_3_TARGETS = {
    "Table 3 #1 (Nomic Baseline, Full Description)": {
        "recall_at_10": 80.28,
        "description": "Baseline nomic-embed-text-v1.5 on full MITRE descriptions",
    },
    "Table 3 #7 (ATTACK-BERT)": {
        "recall_at_10": 86.05,
        "description": "Domain-adapted ATTACK-BERT retrieval",
    },
    "Table 3 #10 (Cross-Encoder Reranker top-15)": {
        "recall_at_10": 74.59,
        "description": "Cross-Encoder reranking over top 15 (observed recall degradation)",
    },
    "Table 3 (First Paragraph)": {
        "recall_at_10": 78.45,
        "description": "Nomic embedding on first paragraph of description",
    },
    "Table 3 (Summarized)": {
        "recall_at_10": 77.12,
        "description": "Nomic embedding on summarized descriptions",
    },
    "Table 3 (Chunks 500)": {
        "recall_at_10": 79.10,
        "description": "Technique descriptions chunked to 500 characters",
    },
}


class Table3RetrievalEngine:
    """Retrieval engine for Table 3 experiments using persistent NumPy index."""

    def __init__(
        self,
        kb_path: Path,
        model_name: str = "nomic-embed-text-v1.5",
        desc_variant: str = "full_description",
        device: str = "cpu",
    ):
        self.kb_path = kb_path
        self.model_name = model_name
        self.desc_variant = desc_variant
        self.device = device

        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.techniques_data = data.get("techniques", [])
        self.desc_variants = data.get("description_variants", {})
        self.technique_ids = [t["technique_id"] for t in self.techniques_data]

    def build_documents(self) -> List[str]:
        """Constructs document strings with appropriate prefix and description variant."""
        docs = []
        is_nomic = "nomic" in self.model_name.lower()
        prefix = "search_document: " if is_nomic else ""

        for tid in self.technique_ids:
            v_info = self.desc_variants.get(tid, {})
            name = v_info.get("technique_name", tid)
            tactics = ", ".join(v_info.get("tactics", []))

            if self.desc_variant == "full_description":
                desc = v_info.get("full_description", "")
            elif self.desc_variant == "first_paragraph":
                desc = v_info.get("first_paragraph", "")
            elif self.desc_variant == "summarized":
                desc = v_info.get("summarized", "")
            elif self.desc_variant == "chunks_500":
                chunks = v_info.get("chunks_500", [""])
                desc = chunks[0] if chunks else ""
            elif self.desc_variant == "chunks_1000":
                chunks = v_info.get("chunks_1000", [""])
                desc = chunks[0] if chunks else ""
            else:
                desc = v_info.get("full_description", "")

            doc_text = f"{prefix}Technique: {name} (ID: {tid}). Tactics: {tactics}. Description: {desc}"
            docs.append(doc_text)
        return docs


def evaluate_retrieval_predictions(
    y_true: List[Set[str]],
    retrieved_candidates_list: List[List[str]],
) -> Dict[str, float]:
    """Calculates all Table 3 metrics."""
    r1 = calculate_paper_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=1)
    r3 = calculate_paper_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=3)
    r5 = calculate_paper_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=5)
    r10 = calculate_paper_retrieval_recall_at_k(y_true, retrieved_candidates_list, k=10)
    p1 = calculate_precision_at_k(y_true, retrieved_candidates_list, k=1)
    mrr = calculate_mean_reciprocal_rank(y_true, retrieved_candidates_list)

    return {
        "recall_at_1": round(r1 * 100, 2),
        "recall_at_3": round(r3 * 100, 2),
        "recall_at_5": round(r5 * 100, 2),
        "recall_at_10": round(r10 * 100, 2),
        "precision_at_1": round(p1 * 100, 2),
        "mrr": round(mrr, 4),
    }
