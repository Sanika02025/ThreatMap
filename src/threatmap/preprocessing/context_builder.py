"""
Context window builder for threat intelligence sentences.
Links neighboring sentences to provide contextual grounding for classifiers.
"""

from typing import List, Optional
from threatmap.schemas import ProcessedSentence
from threatmap.preprocessing.sentence_splitter import split_into_sentences


def build_sentence_contexts(
    raw_text: str,
    document_id: str = "DOC001",
    source_filename: Optional[str] = None,
    min_length: int = 5,
) -> List[ProcessedSentence]:
    """
    Splits document into sentences and associates each sentence with
    its immediate previous and following sentences.
    """
    raw_sentences = split_into_sentences(raw_text, min_length=min_length)
    processed_sentences: List[ProcessedSentence] = []

    total = len(raw_sentences)
    for idx, sentence in enumerate(raw_sentences):
        prev_sent = raw_sentences[idx - 1] if idx > 0 else None
        next_sent = raw_sentences[idx + 1] if idx < total - 1 else None

        sent_obj = ProcessedSentence(
            document_id=document_id,
            sentence_id=f"S{idx + 1:03d}",
            sentence_index=idx,
            original_sentence=sentence,
            normalized_sentence=sentence.lower().strip(),
            paragraph_id=None,
            previous_sentence=prev_sent,
            next_sentence=next_sent,
            source_filename=source_filename,
        )
        processed_sentences.append(sent_obj)

    return processed_sentences


def format_sentence_context(
    sentence: ProcessedSentence,
    window_size: int = 1,
) -> str:
    """
    Constructs a structured prompt context representation distinguishing the target
    sentence from its surrounding context.
    """
    parts = []
    if window_size >= 1 and sentence.previous_sentence:
        parts.append(f"[Previous Context]: {sentence.previous_sentence}")

    parts.append(f"[Target Sentence]: {sentence.original_sentence}")

    if window_size >= 1 and sentence.next_sentence:
        parts.append(f"[Following Context]: {sentence.next_sentence}")

    return "\n".join(parts)
