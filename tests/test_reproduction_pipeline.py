"""
Unit tests for RAG-ATT&CK reproduction data loader, parser, prompts, and metrics.
"""

from threatmap.classification.reproduction_parser import parse_reproduction_response, strip_think_block
from threatmap.classification.reproduction_prompts import (
    build_reproduction_prompt,
    FEW_SHOT_CHAT_EXAMPLES,
    format_candidates,
)
from threatmap.data.tram_loader import load_tram_dataset, TRAMRow, TRAMDataset
from threatmap.evaluation.metrics import (
    calculate_paper_retrieval_recall_at_k,
    calculate_precision_at_k,
    calculate_reproduction_metrics,
)


def test_reproduction_parser_regex():
    raw_output = """
    Based on the analysis:
    T1059.001 : PowerShell
    T1078 : Valid Accounts
    Random text with T1082 : System Information Discovery
    """
    valid_50 = {"T1059.001", "T1078"}
    
    # When restricted to valid_50
    valid_preds, unlisted, records = parse_reproduction_response(raw_output, valid_50_classes=valid_50)
    assert valid_preds == ["T1059.001", "T1078"]
    assert "T1082" in unlisted


def test_reproduction_parser_think_tag():
    raw_output_think = """<think>
    The user is asking about T1059.001 and T1078.
    </think>
    T1059.001 : PowerShell
    """
    valid = {"T1059.001", "T1078"}
    
    # Stripped mode (default)
    preds_stripped, _, _ = parse_reproduction_response(
        raw_output_think, valid_50_classes=valid, is_deepseek_reasoning=True, include_think_in_extraction=False
    )
    assert preds_stripped == ["T1059.001"]
    
    # Think-included mode
    preds_with_think, _, _ = parse_reproduction_response(
        raw_output_think, valid_50_classes=valid, is_deepseek_reasoning=True, include_think_in_extraction=True
    )
    assert set(preds_with_think) == {"T1059.001", "T1078"}


def test_reproduction_prompts():
    prompt_base = build_reproduction_prompt("base_llm", "Attacker launched cmd.exe")
    assert "Attacker launched cmd.exe" in prompt_base
    assert "Txxxx : Name" in prompt_base

    prompt_rag = build_reproduction_prompt(
        "base_rag",
        "Attacker dumped memory",
        candidates=[{"technique_id": "T1003.001", "technique_name": "LSASS Memory", "description": "Dumps LSASS"}]
    )
    assert "T1003.001 : LSASS Memory" in prompt_rag

    messages = build_reproduction_prompt(
        "few_shot",
        "Attacker ran script",
        candidates=[{"technique_id": "T1059.001", "technique_name": "PowerShell", "description": "Runs scripts"}]
    )
    assert isinstance(messages, list)
    assert len(messages) >= 6
    assert messages[0]["role"] == "system"


def test_paper_retrieval_metric():
    # 2 sentences:
    # row 0 has labels ["T1059.001", "T1078"] (2 labels)
    # row 1 has labels ["T1003.001"] (1 label)
    # Total true labels = 3
    ground_truth = [["T1059.001", "T1078"], ["T1003.001"]]
    
    # Retrieved top-2:
    # row 0 retrieved ["T1059.001", "T1082"] -> 1 hit
    # row 1 retrieved ["T1003.001", "T1059.001"] -> 1 hit
    # Total TP = 2
    # Recall@2 = 2 / 3 = 0.6667
    retrieved = [
        ["T1059.001", "T1082"],
        ["T1003.001", "T1059.001"]
    ]
    
    recall_at_2 = calculate_paper_retrieval_recall_at_k(ground_truth, retrieved, k=2)
    assert round(recall_at_2, 4) == 0.6667
    
    # Precision@1: row 0 hit (1/1), row 1 hit (1/1) -> 2/2 = 1.0
    p_at_1 = calculate_precision_at_k(ground_truth, retrieved, k=1)
    assert p_at_1 == 1.0


def test_reproduction_metrics_evaluation():
    y_true = [["T1059.001"], ["T1078"], []]
    y_pred = [["T1059.001", "T9999"], ["T1082"], []]
    valid_50 = {"T1059.001", "T1078", "T1082"}
    
    metrics = calculate_reproduction_metrics(y_true, y_pred, valid_50_classes=valid_50)
    assert metrics["unlisted_predictions"] == 1
    assert "micro_f1" in metrics
    assert "macro_f1" in metrics
