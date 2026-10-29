"""
Builds the 50-technique MITRE ATT&CK Knowledge Base strictly from the October 2024
enterprise-attack.json (ATT&CK v16.0) corresponding to the techniques in TRAM v1.3.

Provides multiple description variants for Table 3 experiments:
1. Full description
2. First paragraph
3. Summarized (first 2-3 sentences)
4. Chunked (500 chars and 1000 chars)
"""

import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from threatmap.data.attack_loader import parse_attack_stix_bundle, AttackKnowledgeBase
from threatmap.logging_config import setup_logger
from threatmap.schemas import AttackTechnique

logger = setup_logger("threatmap.build_50_techniques_kb")


def extract_first_paragraph(desc: str) -> str:
    paragraphs = [p.strip() for p in desc.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[0]
    return desc[:300].strip()


def extract_summary(desc: str) -> str:
    # First 2 sentences
    sentences = re.split(r"(?<=[.!?])\s+", desc.strip())
    if len(sentences) >= 2:
        return " ".join(sentences[:2]).strip()
    return desc[:250].strip()


def chunk_text(text: str, chunk_size: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += chunk_size
    return chunks if chunks else [text]


def build_50_techniques_kb(
    stix_path: Path,
    tram_path: Path,
    output_path: Path,
) -> AttackKnowledgeBase:
    logger.info(f"Loading TRAM dataset from {tram_path} to extract the 50 target classes...")
    with open(tram_path, "r", encoding="utf-8") as f:
        tram_data = json.load(f)

    target_50_ids = sorted(list(set(lbl for d in tram_data for lbl in d.get("labels", []))))
    logger.info(f"Target technique classes ({len(target_50_ids)}): {target_50_ids}")

    logger.info(f"Loading and parsing MITRE STIX v16.0 from {stix_path}...")
    with open(stix_path, "r", encoding="utf-8") as f:
        stix_data = json.load(f)

    full_kb = parse_attack_stix_bundle(stix_data)
    logger.info(f"Parsed {len(full_kb)} total techniques from STIX.")

    techniques_50: List[AttackTechnique] = []
    descriptions_metadata: Dict[str, Dict[str, Any]] = {}

    for tid in target_50_ids:
        tech = full_kb.get_technique(tid)
        if not tech:
            raise ValueError(f"Technique {tid} not found in STIX v16 dataset!")

        techniques_50.append(tech)

        full_desc = tech.description
        p1_desc = extract_first_paragraph(full_desc)
        summary_desc = extract_summary(full_desc)
        chunks_500 = chunk_text(full_desc, 500)
        chunks_1000 = chunk_text(full_desc, 1000)

        descriptions_metadata[tid] = {
            "technique_id": tid,
            "technique_name": tech.technique_name,
            "tactics": tech.tactic_names,
            "full_description": full_desc,
            "first_paragraph": p1_desc,
            "summarized": summary_desc,
            "chunks_500": chunks_500,
            "chunks_1000": chunks_1000,
        }

    kb_50 = AttackKnowledgeBase(
        techniques=techniques_50,
        source_info="MITRE ATT&CK Enterprise v16.0 (October 2024) - Restricted to 50 TRAM Reproduction Classes",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_info": kb_50.source_info,
        "count": len(techniques_50),
        "target_techniques_50": target_50_ids,
        "techniques": kb_50.to_dict_list(),
        "description_variants": descriptions_metadata,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully saved 50-technique knowledge base to {output_path}")
    return kb_50


def main():
    stix_file = BASE_DIR / "data" / "raw" / "enterprise-attack-v16.json"
    tram_file = BASE_DIR / "data" / "raw" / "tram_multi_label.json"
    out_file = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"

    build_50_techniques_kb(stix_file, tram_file, out_file)


if __name__ == "__main__":
    main()
