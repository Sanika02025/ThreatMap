"""
Unit tests for BM25 lexical and semantic vector retrieval.
"""

from threatmap.data.attack_loader import load_attack_data
from threatmap.retrieval.lexical import BM25Retriever
from threatmap.retrieval.semantic import SemanticRetriever


def test_bm25_retrieval_exact_match():
    kb = load_attack_data(fallback_to_fixture=True)
    techniques = kb.get_all_active_techniques()
    retriever = BM25Retriever(techniques)

    # Query specifically mentioning PowerShell and T1059.001
    results = retriever.retrieve("The malware invoked a PowerShell command T1059.001", top_k=3)
    assert len(results) > 0
    top = results[0]
    assert top.technique_id == "T1059.001"
    assert top.lexical_score > 0.0


def test_bm25_retrieval_empty_query():
    kb = load_attack_data(fallback_to_fixture=True)
    retriever = BM25Retriever(kb.get_all_active_techniques())
    assert retriever.retrieve("", top_k=5) == []


def test_semantic_retrieval():
    kb = load_attack_data(fallback_to_fixture=True)
    techniques = kb.get_all_active_techniques()

    retriever = SemanticRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    retriever.index_techniques(techniques)
    assert retriever.embeddings is not None
    assert retriever.embeddings.shape[0] == len(techniques)

    # Semantic query without exact keyword matches
    results = retriever.retrieve("Adversary extracted stored credentials from LSASS memory", top_k=3)
    assert len(results) == 3
    top_ids = [r.technique_id for r in results]
    assert "T1003.001" in top_ids
    assert results[0].semantic_score is not None
