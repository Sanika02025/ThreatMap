"""
Pydantic schemas and data contracts for ThreatMap.
Ensures strict validation, consistent interfaces, and typed serializability.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AttackTechnique(BaseModel):
    """Normalized MITRE ATT&CK Technique representation."""
    technique_id: str = Field(..., description="Unique ATT&CK ID, e.g., T1059.001")
    technique_name: str = Field(..., description="Technique name, e.g., PowerShell")
    description: str = Field(..., description="Full or normalized ATT&CK description")
    tactic_names: List[str] = Field(default_factory=list, description="Tactics associated with this technique")
    tactic_ids: List[str] = Field(default_factory=list, description="Tactic IDs (e.g. TA0002)")
    platforms: List[str] = Field(default_factory=list, description="Supported platforms, e.g., Windows, Linux")
    detection: Optional[str] = Field(None, description="Detection guidance if available")
    data_sources: List[str] = Field(default_factory=list, description="Relevant data sources")
    source_url: Optional[str] = Field(None, description="Official MITRE ATT&CK URL")
    is_subtechnique: bool = Field(False, description="Whether this is a sub-technique")
    subtechnique_of: Optional[str] = Field(None, description="Parent technique ID if sub-technique")
    revoked: bool = Field(False, description="Whether technique is revoked")
    deprecated: bool = Field(False, description="Whether technique is deprecated")
    version: Optional[str] = Field(None, description="ATT&CK specification version")


class ProcessedSentence(BaseModel):
    """Segmented sentence enriched with contextual neighbors."""
    document_id: str
    sentence_id: str
    sentence_index: int
    original_sentence: str
    normalized_sentence: str
    paragraph_id: Optional[int] = None
    previous_sentence: Optional[str] = None
    next_sentence: Optional[str] = None
    source_filename: Optional[str] = None


class RetrievedCandidate(BaseModel):
    """Retrieved candidate technique from lexical, semantic, or hybrid search."""
    technique_id: str
    technique_name: str
    tactics: List[str] = Field(default_factory=list)
    description: str
    lexical_rank: Optional[int] = None
    lexical_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    fused_rank: int
    fused_score: float
    rerank_score: Optional[float] = None
    retrieval_method: str = "hybrid"


class TechniquePrediction(BaseModel):
    """Final technique prediction with evidence grounding."""
    technique_id: str
    technique_name: str
    tactic: Optional[str] = None
    evidence_quote: str
    explanation: str
    evidence_type: str = "explicit"  # "explicit", "inferred", "weak"
    retrieval_score: float = 0.0
    llm_confidence: Optional[float] = None


class SentenceAnalysisResult(BaseModel):
    """Complete analysis output for a single sentence."""
    sentence_id: str
    original_sentence: str
    context_text: Optional[str] = None
    predictions: List[TechniquePrediction] = Field(default_factory=list)
    alternative_candidates: List[RetrievedCandidate] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: Optional[str] = None
    validated_by_llm: bool = False


class ReportAnalysisResult(BaseModel):
    """Aggregated analysis output for a complete threat report."""
    document_id: str
    filename: Optional[str] = None
    timestamp: str
    total_sentences: int
    sentences_with_predictions: int
    abstention_count: int
    unique_techniques: List[str] = Field(default_factory=list)
    tactic_distribution: Dict[str, int] = Field(default_factory=dict)
    sentence_results: List[SentenceAnalysisResult] = Field(default_factory=list)
    pipeline_metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkExample(BaseModel):
    """Ground truth benchmark example for evaluation."""
    id: str
    text: str
    labels: List[str] = Field(default_factory=list)
    split: Optional[str] = "test"
    source_report: Optional[str] = None


class EvaluationMetrics(BaseModel):
    """Comprehensive multilabel evaluation metrics."""
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    coverage: float
    abstention_rate: float
    per_technique: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_examples: int
    evaluated_techniques_count: int
    runtime_seconds: float = 0.0
