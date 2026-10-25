"""
Lexical retrieval engine based on BM25 Okapi.
Optimized for exact keyword matches, technique identifiers, and security tool names.
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi
from threatmap.logging_config import setup_logger
from threatmap.schemas import AttackTechnique, RetrievedCandidate

logger = setup_logger("threatmap.lexical")

# Tokenization regex keeping dots, dashes, and underscores in technical identifiers
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\.\-]+")


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text while preserving identifiers (e.g. T1059.001, powershell.exe, lsass)."""
    if not text:
        return []
    tokens = TOKEN_PATTERN.findall(text.lower())
    # Filter pure single punctuation or empty tokens
    return [t for t in tokens if len(t) > 1 or t.isalnum()]


class BM25Retriever:
    """BM25 Okapi retriever for ATT&CK techniques."""

    def __init__(self, techniques: Optional[List[AttackTechnique]] = None):
        self.techniques: List[AttackTechnique] = []
        self.technique_map: Dict[str, AttackTechnique] = {}
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if techniques:
            self.index_techniques(techniques)

    def _build_technique_document(self, t: AttackTechnique) -> str:
        """Constructs an information-dense text representation for BM25 indexing."""
        tactics_str = " ".join(t.tactic_names)
        platforms_str = " ".join(t.platforms)
        data_sources_str = " ".join(t.data_sources)
        # Give higher weight to ID and Name by repeating them in the document text
        doc = f"{t.technique_id} {t.technique_name} {t.technique_id} {t.technique_name} {tactics_str} {platforms_str} {data_sources_str} {t.description}"
        return doc

    def index_techniques(self, techniques: List[AttackTechnique]) -> None:
        """Indexes active techniques with BM25."""
        self.techniques = [t for t in techniques if not t.revoked]
        self.technique_map = {t.technique_id: t for t in self.techniques}
        self.corpus_tokens = [
            tokenize_for_bm25(self._build_technique_document(t)) for t in self.techniques
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"Indexed {len(self.techniques)} techniques into BM25 retriever.")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedCandidate]:
        """Retrieves top-k candidates using BM25 scoring."""
        if not self.bm25 or not self.techniques:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        # Rank indices descending by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        candidates: List[RetrievedCandidate] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                continue
            tech = self.techniques[idx]
            candidates.append(
                RetrievedCandidate(
                    technique_id=tech.technique_id,
                    technique_name=tech.technique_name,
                    tactics=tech.tactic_names,
                    description=tech.description,
                    lexical_rank=rank,
                    lexical_score=round(score, 4),
                    semantic_rank=None,
                    semantic_score=None,
                    fused_rank=rank,
                    fused_score=round(score, 4),
                    retrieval_method="lexical",
                )
            )

        return candidates

    def save(self, file_path: str | Path) -> None:
        """Serializes retriever to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "techniques": self.techniques,
                    "corpus_tokens": self.corpus_tokens,
                    "bm25": self.bm25,
                },
                f,
            )
        logger.info(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25Retriever":
        """Loads serialized BM25 retriever from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        retriever = cls()
        retriever.techniques = data["techniques"]
        retriever.technique_map = {t.technique_id: t for t in retriever.techniques}
        retriever.corpus_tokens = data["corpus_tokens"]
        retriever.bm25 = data["bm25"]
        return retriever
