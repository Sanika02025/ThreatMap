"""
Builds persistent BM25 lexical and SentenceTransformer vector indices.
Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from threatmap.config import load_config
from threatmap.logging_config import setup_logger
from threatmap.retrieval.index_manager import IndexManager

logger = setup_logger("threatmap.build_index")


def main():
    logger.info("Initializing Index Manager...")
    config = load_config(base_dir=BASE_DIR)
    manager = IndexManager(config, base_dir=BASE_DIR)

    logger.info("Building indices (force_rebuild=True)...")
    retriever = manager.build_indices(force_rebuild=True)
    logger.info("Index build completed successfully!")
    logger.info(f"Vector Index Path: {manager.vector_index_path}")
    logger.info(f"BM25 Index Path: {manager.bm25_index_path}")


if __name__ == "__main__":
    main()
