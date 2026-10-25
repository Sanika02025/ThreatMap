"""
Configuration loader and validator for ThreatMap.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_attack_dir: str = "data/raw"
    processed_attack_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    evaluation_dir: str = "data/evaluation"
    samples_dir: str = "data/samples"
    attack_stix_url: str = (
        "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    )
    raw_attack_file: str = "data/raw/enterprise-attack.json"
    processed_attack_file: str = "data/processed/mitre_attack_enterprise.json"
    vector_index_file: str = "artifacts/vector_index.npy"
    index_metadata_file: str = "artifacts/index_metadata.json"
    bm25_index_file: str = "artifacts/bm25_index.pkl"


class RetrievalConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    top_k: int = 5
    rrf_k: int = 60
    hybrid_mode: str = "rrf"
    weighted_lexical_weight: float = 0.5
    weighted_semantic_weight: float = 0.5
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PreprocessingConfig(BaseModel):
    context_window_size: int = 1
    min_sentence_length: int = 5
    preserve_technical_tokens: bool = True


class ClassificationConfig(BaseModel):
    enabled: bool = False
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: int = 30
    context_strategy: str = "target_with_context_and_candidates"


class AbstentionConfig(BaseModel):
    min_semantic_similarity: float = 0.30
    min_rrf_score: float = 0.012
    require_valid_attack_id: bool = True
    min_evidence_char_length: int = 10
    verify_evidence_in_source: bool = True


class EvaluationConfig(BaseModel):
    benchmark_dataset: str = "data/evaluation/tram_benchmark.json"
    zero_division_strategy: float = 0.0


class ThreatMapConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(config_path: Optional[str] = None, base_dir: Optional[Path] = None) -> ThreatMapConfig:
    """Loads configuration from YAML file or returns default."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent

    target_path = Path(config_path) if config_path else base_dir / "configs" / "default.yaml"

    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
            return ThreatMapConfig(**raw_data)

    return ThreatMapConfig()
