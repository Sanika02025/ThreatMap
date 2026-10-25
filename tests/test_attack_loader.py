"""
Unit tests for MITRE ATT&CK knowledge base loading, STIX parsing, and validation.
"""

from threatmap.data.attack_loader import (
    AttackKnowledgeBase,
    load_attack_data,
    parse_attack_stix_bundle,
)
from threatmap.data.validators import is_valid_attack_id_format, validate_technique_id


def test_is_valid_attack_id_format():
    assert is_valid_attack_id_format("T1059") is True
    assert is_valid_attack_id_format("T1059.001") is True
    assert is_valid_attack_id_format("T1053.005") is True
    assert is_valid_attack_id_format("T105") is False
    assert is_valid_attack_id_format("PowerShell") is False
    assert is_valid_attack_id_format("1059") is False
    assert is_valid_attack_id_format("") is False


def test_load_attack_data_fixture_fallback():
    kb = load_attack_data(fallback_to_fixture=True)
    assert len(kb) > 10
    tech = kb.get_technique("T1059.001")
    assert tech is not None
    assert tech.technique_name == "Command and Scripting Interpreter: PowerShell"
    assert "Execution" in tech.tactic_names
    assert tech.is_subtechnique is True
    assert tech.subtechnique_of == "T1059"


def test_validate_technique_id_in_kb():
    kb = load_attack_data(fallback_to_fixture=True)
    valid_ids = kb.get_technique_ids()
    assert validate_technique_id("T1059.001", valid_ids) is True
    assert validate_technique_id("T9999", valid_ids) is False
    assert validate_technique_id("INVALID", valid_ids) is False


def test_parse_attack_stix_bundle_synthetic():
    synthetic_bundle = {
        "spec_version": "2.1",
        "objects": [
            {
                "type": "attack-pattern",
                "id": "attack-pattern--1",
                "name": "PowerShell Execution",
                "description": "Adversaries execute powershell scripts.",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"}
                ],
                "kill_chain_phases": [
                    {"kill_chain_name": "mitre-attack", "phase_name": "execution"}
                ],
                "x_mitre_is_subtechnique": True,
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--2",
                "name": "Command and Scripting Interpreter",
                "description": "Parent technique.",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059"}
                ],
                "kill_chain_phases": [
                    {"kill_chain_name": "mitre-attack", "phase_name": "execution"}
                ],
                "x_mitre_is_subtechnique": False,
            },
            {
                "type": "relationship",
                "id": "relationship--1",
                "relationship_type": "subtechnique-of",
                "source_ref": "attack-pattern--1",
                "target_ref": "attack-pattern--2",
            },
        ],
    }
    kb = parse_attack_stix_bundle(synthetic_bundle)
    assert len(kb) == 2
    subtech = kb.get_technique("T1059.001")
    assert subtech is not None
    assert subtech.subtechnique_of == "T1059"
    assert "Execution" in subtech.tactic_names
