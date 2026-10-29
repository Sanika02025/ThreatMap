"""
Semantic retrieval engine using Sentence Transformers and local vector indexing.
Uses cosine similarity over normalized embeddings with offline fallback.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from threatmap.logging_config import setup_logger
from threatmap.schemas import AttackTechnique, RetrievedCandidate

logger = setup_logger("threatmap.semantic")


class OfflineFallbackEmbedder:
    """Lightweight offline dense embedder using TF-IDF feature hashing when hub is offline."""

    def __init__(self, max_features: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True)
        self.is_fitted = False

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if not self.is_fitted:
            mat = self.vectorizer.fit_transform(texts).toarray()
            self.is_fitted = True
        else:
            mat = self.vectorizer.transform(texts).toarray()

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)


class SemanticRetriever:
    """Vector index and semantic search over ATT&CK techniques."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None  # Lazy loaded
        self.techniques: List[AttackTechnique] = []
        self.technique_map: Dict[str, AttackTechnique] = {}
        self.embeddings: Optional[np.ndarray] = None  # Normalized float32 shape (N, D)

    @property
    def model(self):
        """Lazy loads SentenceTransformer model with fallback to offline embedder."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name} on {self.device}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer from remote hub ({e}). Falling back to offline dense embedder.")
                self._model = OfflineFallbackEmbedder()
        return self._model

    def _build_technique_document(self, t: AttackTechnique) -> str:
        """Constructs an information-dense semantic document for embedding."""
        tactics_str = ", ".join(t.tactic_names) if t.tactic_names else "General"
        platforms_str = ", ".join(t.platforms) if t.platforms else "All"
        return (
            f"Technique: {t.technique_name} (ID: {t.technique_id}). "
            f"Tactics: {tactics_str}. Platforms: {platforms_str}. "
            f"Description: {t.description}"
        )

    def index_techniques(self, techniques: List[AttackTechnique]) -> None:
        """Computes and normalizes embeddings for active techniques."""
        self.techniques = [t for t in techniques if not t.revoked]
        self.technique_map = {t.technique_id: t for t in self.techniques}

        texts = [self._build_technique_document(t) for t in self.techniques]
        logger.info(f"Encoding {len(texts)} ATT&CK techniques with {self.model_name}...")

        raw_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Ensures unit vectors for exact dot-product cosine similarity
        )
        self.embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        logger.info(f"Generated semantic index with shape {self.embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedCandidate]:
        """Retrieves top-k techniques based on cosine similarity."""
        if self.embeddings is None or not self.techniques:
            return []

        query_text = query.strip()
        if not query_text:
            return []

        raw_q_vec = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vec = np.asarray(raw_q_vec, dtype=np.float32)[0]

        # Dot product of normalized vectors equals cosine similarity
        similarities = np.dot(self.embeddings, query_vec)

        # Top-k indices sorted descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates: List[RetrievedCandidate] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            tech = self.techniques[idx]
            candidates.append(
                RetrievedCandidate(
                    technique_id=tech.technique_id,
                    technique_name=tech.technique_name,
                    tactics=tech.tactic_names,
                    description=tech.description,
                    lexical_rank=None,
                    lexical_score=None,
                    semantic_rank=rank,
                    semantic_score=round(score, 4),
                    fused_rank=rank,
                    fused_score=round(score, 4),
                    retrieval_method="semantic",
                )
            )

        return candidates

    def save(self, index_file: str | Path, metadata_file: str | Path) -> None:
        """Saves embedding array and metadata to disk."""
        if self.embeddings is None:
            raise ValueError("No embeddings computed to save.")

        idx_path = Path(index_file)
        meta_path = Path(metadata_file)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(idx_path, self.embeddings)

        meta = {
            "model_name": self.model_name,
            "dimension": int(self.embeddings.shape[1]),
            "count": int(self.embeddings.shape[0]),
            "techniques": [t.model_dump() for t in self.techniques],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved vector index to {idx_path} and metadata to {meta_path}")

    @classmethod
    def load(
        cls,
        index_file: str | Path,
        metadata_file: str | Path,
        device: str = "cpu",
    ) -> "SemanticRetriever":
        """Loads vector index and metadata from disk without re-encoding."""
        idx_path = Path(index_file)
        meta_path = Path(metadata_file)

        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Vector index or metadata not found at {idx_path}, {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_name = meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        retriever = cls(model_name=model_name, device=device)
        retriever.embeddings = np.load(idx_path).astype(np.float32)
        retriever.techniques = [AttackTechnique(**t) for t in meta["techniques"]]
        retriever.technique_map = {t.technique_id: t for t in retriever.techniques}

        logger.info(f"Loaded vector index ({retriever.embeddings.shape}) with {len(retriever.techniques)} techniques")
        return retriever
