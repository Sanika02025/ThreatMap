"""
Index manager to build, save, load, and coordinate lexical and semantic indexes.
"""

from pathlib import Path
from typing import Optional
from threatmap.config import ThreatMapConfig
from threatmap.data.attack_loader import AttackKnowledgeBase, load_attack_data
from threatmap.logging_config import setup_logger
from threatmap.retrieval.hybrid import HybridRetriever
from threatmap.retrieval.lexical import BM25Retriever
from threatmap.retrieval.semantic import SemanticRetriever

logger = setup_logger("threatmap.index_manager")


class IndexManager:
    """Orchestrates persistent lexical and semantic vector indices."""

    def __init__(self, config: ThreatMapConfig, base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent.parent

        self.vector_index_path = self.base_dir / config.paths.vector_index_file
        self.index_meta_path = self.base_dir / config.paths.index_metadata_file
        self.bm25_index_path = self.base_dir / config.paths.bm25_index_file
        self.processed_attack_path = self.base_dir / config.paths.processed_attack_file
        self.raw_attack_path = self.base_dir / config.paths.raw_attack_file

    def indices_exist(self) -> bool:
        """Checks if both vector and BM25 persistent index files exist."""
        return (
            self.vector_index_path.exists()
            and self.index_meta_path.exists()
            and self.bm25_index_path.exists()
        )

    def build_indices(
        self,
        kb: Optional[AttackKnowledgeBase] = None,
        force_rebuild: bool = False,
    ) -> HybridRetriever:
        """
        Builds both lexical and semantic indices from the knowledge base
        and saves them to disk.
        """
        if not force_rebuild and self.indices_exist():
            logger.info("Existing indices found on disk. Loading without rebuilding...")
            return self.load_retriever()

        if kb is None:
            kb = load_attack_data(
                processed_path=self.processed_attack_path,
                raw_path=self.raw_attack_path,
                fallback_to_fixture=True,
            )

        active_techniques = kb.get_all_active_techniques(include_deprecated=False)
        logger.info(f"Building indices for {len(active_techniques)} active ATT&CK techniques...")

        # 1. Build and save BM25
        bm25_retriever = BM25Retriever(active_techniques)
        bm25_retriever.save(self.bm25_index_path)

        # 2. Build and save Semantic Index
        sem_retriever = SemanticRetriever(
            model_name=self.config.retrieval.embedding_model,
            device=self.config.retrieval.device,
            batch_size=self.config.retrieval.batch_size,
        )
        sem_retriever.index_techniques(active_techniques)
        sem_retriever.save(self.vector_index_path, self.index_meta_path)

        return HybridRetriever(
            bm25_retriever=bm25_retriever,
            semantic_retriever=sem_retriever,
            default_mode=self.config.retrieval.hybrid_mode,
            rrf_k=self.config.retrieval.rrf_k,
        )

    def load_retriever(self) -> HybridRetriever:
        """Loads persistent retriever from disk. Falls back to building if missing."""
        if not self.indices_exist():
            logger.info("Indices missing from disk. Auto-building...")
            return self.build_indices()

        bm25_retriever = BM25Retriever.load(self.bm25_index_path)
        sem_retriever = SemanticRetriever.load(
            index_file=self.vector_index_path,
            metadata_file=self.index_meta_path,
            device=self.config.retrieval.device,
        )

        return HybridRetriever(
            bm25_retriever=bm25_retriever,
            semantic_retriever=sem_retriever,
            default_mode=self.config.retrieval.hybrid_mode,
            rrf_k=self.config.retrieval.rrf_k,
        )
