"""
Unit tests for document parsing, cyber-aware sentence splitting, and context window generation.
"""

import pytest
from threatmap.preprocessing.document_parser import parse_document
from threatmap.preprocessing.sentence_splitter import split_into_sentences
from threatmap.preprocessing.context_builder import build_sentence_contexts, format_sentence_context


def test_split_into_sentences_preserves_technical_tokens():
    text = (
        "The attacker executed powershell.exe on host 192.168.1.5 to fetch payload.dll. "
        "Adversary exploited CVE-2023-38831 at https://malicious.example.com/test.php. "
        "Subtechnique T1059.001 was observed using v1.3.2."
    )
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert "powershell.exe" in sentences[0]
    assert "192.168.1.5" in sentences[0]
    assert "payload.dll" in sentences[0]
    assert "CVE-2023-38831" in sentences[1]
    assert "https://malicious.example.com/test.php" in sentences[1]
    assert "T1059.001" in sentences[2]
    assert "v1.3.2" in sentences[2]


def test_split_into_sentences_empty_and_whitespace():
    assert split_into_sentences("") == []
    assert split_into_sentences("   \n\n\t  ") == []


def test_split_into_sentences_abbreviations():
    text = "Adversaries abused living-off-the-land binaries, e.g. certutil.exe, to stage files. Next sentence."
    sentences = split_into_sentences(text)
    assert len(sentences) == 2
    assert "e.g." in sentences[0]


def test_parse_document_plain_text():
    raw = "Adversary gained initial access via spearphishing attachment."
    doc_id, text = parse_document(raw)
    assert len(doc_id) == 12
    assert text == raw


def test_parse_document_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_document("")


def test_build_sentence_contexts():
    text = "First sentence of report. Second sentence describes execution. Third sentence describes exfiltration."
    sentences = build_sentence_contexts(text, document_id="DOC_TEST")
    assert len(sentences) == 3
    assert sentences[0].sentence_id == "S001"
    assert sentences[0].previous_sentence is None
    assert sentences[0].next_sentence == "Second sentence describes execution."

    assert sentences[1].sentence_id == "S002"
    assert sentences[1].previous_sentence == "First sentence of report."
    assert sentences[1].next_sentence == "Third sentence describes exfiltration."

    formatted = format_sentence_context(sentences[1], window_size=1)
    assert "[Previous Context]: First sentence of report." in formatted
    assert "[Target Sentence]: Second sentence describes execution." in formatted
    assert "[Following Context]: Third sentence describes exfiltration." in formatted
