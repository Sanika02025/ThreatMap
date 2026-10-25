"""
End-to-end ThreatMap Pipeline orchestrator.
Coordinates preprocessing, retrieval, optional reranking, local LLM classification,
evidence validation, and abstention control.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from threatmap.config import ThreatMapConfig
from threatmap.data.attack_loader import AttackKnowledgeBase, load_attack_data
from threatmap.logging_config import setup_logger
from threatmap.preprocessing.context_builder import build_sentence_contexts, format_sentence_context
from threatmap.preprocessing.document_parser import parse_document
from threatmap.retrieval.hybrid import HybridRetriever
from threatmap.retrieval.index_manager import IndexManager
from threatmap.retrieval.reranker import CrossEncoderReranker
from threatmap.schemas import (
    ProcessedSentence,
    ReportAnalysisResult,
    RetrievedCandidate,
    SentenceAnalysisResult,
    TechniquePrediction,
)
from threatmap.classification.abstention import AbstentionEngine
from threatmap.classification.evidence_validator import validate_prediction_evidence
from threatmap.classification.llm_client import OllamaClient
from threatmap.classification.prediction_parser import parse_llm_response
from threatmap.classification.prompt_builder import SYSTEM_PROMPT, build_classification_prompt

logger = setup_logger("threatmap.pipeline")


class ThreatMapPipeline:
    """End-to-end processing pipeline for cyber threat intelligence reports."""

    def __init__(
        self,
        config: ThreatMapConfig,
        kb: Optional[AttackKnowledgeBase] = None,
        retriever: Optional[HybridRetriever] = None,
        base_dir: Optional[Path] = None,
    ):
        self.config = config
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent.parent

        # 1. Knowledge Base
        self.kb = kb or load_attack_data(
            processed_path=self.base_dir / config.paths.processed_attack_file,
            raw_path=self.base_dir / config.paths.raw_attack_file,
            fallback_to_fixture=True,
        )

        # 2. Retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            mgr = IndexManager(config, base_dir=self.base_dir)
            self.retriever = mgr.build_indices(kb=self.kb)

        # 3. Optional Reranker
        self.reranker = CrossEncoderReranker(
            model_name=config.retrieval.reranker_model,
            enabled=config.retrieval.use_reranker,
            device=config.retrieval.device,
        )

        # 4. Optional Local LLM
        self.llm_client = OllamaClient(
            base_url=config.classification.ollama_base_url,
            model_name=config.classification.ollama_model,
            temperature=config.classification.temperature,
            timeout_seconds=config.classification.timeout_seconds,
        )

        # 5. Abstention Engine
        self.abstention_engine = AbstentionEngine(config.abstention)

    def analyze_sentence(
        self,
        sentence: ProcessedSentence,
        retrieval_mode: str = "hybrid_rrf",
        top_k: int = 5,
        use_llm: bool = False,
        context_strategy: str = "target_with_context_and_candidates",
    ) -> SentenceAnalysisResult:
        """Processes a single sentence through retrieval, classification, and validation."""
        # 1. Retrieval
        candidates: List[RetrievedCandidate] = self.retriever.retrieve(
            query=sentence.original_sentence,
            mode=retrieval_mode,
            top_k=top_k,
        )

        # 2. Optional Reranking
        if self.reranker.enabled:
            candidates = self.reranker.rerank(
                query=sentence.original_sentence,
                candidates=candidates,
                top_k=top_k,
            )

        # Format context text for auditability
        context_text = format_sentence_context(
            sentence, window_size=self.config.preprocessing.context_window_size
        )

        # 3. Classification stage (LLM or Retrieval-Only Fallback)
        validated_predictions: List[TechniquePrediction] = []
        rejection_reasons: List[str] = []
        is_llm_validated = False

        if use_llm and self.llm_client.is_available():
            prompt = build_classification_prompt(
                sentence=sentence,
                candidates=candidates,
                context_strategy=context_strategy,
                window_size=self.config.preprocessing.context_window_size,
            )
            raw_response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                json_format=True,
            )
            parsed_result = parse_llm_response(raw_response, default_sentence_id=sentence.sentence_id)

            if not parsed_result.get("abstained", False):
                raw_preds = parsed_result.get("predictions", [])
                validated_predictions, rejection_reasons = validate_prediction_evidence(
                    raw_predictions=raw_preds,
                    source_sentence=sentence.original_sentence,
                    context_sentence=context_text,
                    kb=self.kb,
                    require_valid_id=self.config.abstention.require_valid_attack_id,
                    verify_quote=self.config.abstention.verify_evidence_in_source,
                )
                if validated_predictions:
                    is_llm_validated = True

        else:
            # Retrieval-only candidate mapping mode
            # In absence of LLM, the top candidate is considered if it passes confidence thresholds
            if candidates:
                top_cand = candidates[0]
                tech = self.kb.get_technique(top_cand.technique_id)
                tactic = tech.tactic_names[0] if (tech and tech.tactic_names) else "General"

                cand_pred = {
                    "technique_id": top_cand.technique_id,
                    "technique_name": top_cand.technique_name,
                    "tactic": tactic,
                    "evidence_quote": sentence.original_sentence,
                    "explanation": f"Top candidate retrieved via {retrieval_mode} ranking (score: {top_cand.fused_score}).",
                    "evidence_type": "retrieved_candidate",
                    "retrieval_score": top_cand.fused_score,
                    "llm_confidence": None,
                }
                validated_predictions, rejection_reasons = validate_prediction_evidence(
                    raw_predictions=[cand_pred],
                    source_sentence=sentence.original_sentence,
                    context_sentence=context_text,
                    kb=self.kb,
                    require_valid_id=True,
                    verify_quote=False,
                )

        # 4. Abstention decision
        should_abstain, abstention_reason = self.abstention_engine.should_abstain(
            sentence_text=sentence.original_sentence,
            candidates=candidates,
            validated_predictions=validated_predictions,
            rejection_reasons=rejection_reasons,
            retrieval_mode=retrieval_mode,
        )

        final_predictions = [] if should_abstain else validated_predictions

        return SentenceAnalysisResult(
            sentence_id=sentence.sentence_id,
            original_sentence=sentence.original_sentence,
            context_text=context_text,
            predictions=final_predictions,
            alternative_candidates=candidates,
            abstained=should_abstain,
            abstention_reason=abstention_reason,
            validated_by_llm=is_llm_validated,
        )

    def analyze_report(
        self,
        content: Union[str, bytes, Path],
        filename: Optional[str] = None,
        retrieval_mode: str = "hybrid_rrf",
        top_k: int = 5,
        use_llm: bool = False,
        context_strategy: str = "target_with_context_and_candidates",
    ) -> ReportAnalysisResult:
        """Processes an entire threat report document."""
        doc_id, raw_text = parse_document(content, filename=filename)
        sentences = build_sentence_contexts(
            raw_text=raw_text,
            document_id=doc_id,
            source_filename=filename,
            min_length=self.config.preprocessing.min_sentence_length,
        )

        sentence_results: List[SentenceAnalysisResult] = []
        unique_techniques: Set = set()
        tactic_distribution: Dict[str, int] = {}
        abstention_count = 0
        predicted_sentences_count = 0

        for sent in sentences:
            res = self.analyze_sentence(
                sentence=sent,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                use_llm=use_llm,
                context_strategy=context_strategy,
            )
            sentence_results.append(res)

            if res.abstained:
                abstention_count += 1
            else:
                if res.predictions:
                    predicted_sentences_count += 1
                    for pred in res.predictions:
                        unique_techniques.add(pred.technique_id)
                        tactic = pred.tactic or "Unknown"
                        tactic_distribution[tactic] = tactic_distribution.get(tactic, 0) + 1

        return ReportAnalysisResult(
            document_id=doc_id,
            filename=filename,
            timestamp=datetime.now().isoformat(),
            total_sentences=len(sentences),
            sentences_with_predictions=predicted_sentences_count,
            abstention_count=abstention_count,
            unique_techniques=sorted(list(unique_techniques)),
            tactic_distribution=tactic_distribution,
            sentence_results=sentence_results,
            pipeline_metadata={
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "use_llm": use_llm,
                "llm_available": self.llm_client.is_available(),
                "embedding_model": self.config.retrieval.embedding_model,
                "reranker_enabled": self.reranker.enabled,
                "context_strategy": context_strategy,
                "kb_techniques_count": len(self.kb),
            },
        )
