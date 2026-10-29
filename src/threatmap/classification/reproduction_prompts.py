"""
Verbatim prompt templates corresponding to RAG-ATT&CK (Schuurman, 2025) Appendices B–F.
Prompts generate plain text 'Txxxx : Name' responses parsed via regular expressions.
"""

from typing import Dict, List, Optional, Union
from threatmap.schemas import RetrievedCandidate

# Appendix B: BaseLLM (Zero-context prompt)
APPENDIX_B_BASE_LLM = """You are a cybersecurity expert. Given a sentence from a Cyber Threat Intelligence (CTI) report, identify the relevant MITRE ATT&CK techniques.

Provide your answer as a list of technique identifiers and names in the format:
Txxxx : Name
If no techniques apply, output 'None'.

Sentence: {sentence}"""

# Appendix C: BaseRAG (Retrieval-Augmented Prompt with candidates)
APPENDIX_C_BASE_RAG = """You are a cybersecurity expert. Given a sentence from a Cyber Threat Intelligence (CTI) report and candidate MITRE ATT&CK techniques, identify which technique(s) apply.

Candidate Techniques:
{candidates_formatted}

Provide your answer as a list of technique identifiers and names in the format:
Txxxx : Name
If no techniques apply, output 'None'.

Sentence: {sentence}"""

# Appendix D: Optimized RAG (Configurable k candidates with structured descriptions)
APPENDIX_D_OPTIMIZED_RAG = """You are a cybersecurity expert. Given a sentence from a Cyber Threat Intelligence (CTI) report and top-{k} candidate MITRE ATT&CK techniques, identify which technique(s) are described in the sentence.

Candidate Techniques:
{candidates_formatted}

Rules:
- Select only technique(s) that are directly evidenced in the sentence.
- Provide your answer as a list of technique identifiers and names in the format:
Txxxx : Name
- If no techniques apply, output 'None'.

Sentence: {sentence}"""

# Appendix E: Context Prompt (Incorporating 1 or 2 neighboring sentences)
APPENDIX_E_CONTEXT = """You are a cybersecurity expert. Given a target sentence and its surrounding context from a Cyber Threat Intelligence (CTI) report along with candidate MITRE ATT&CK techniques, identify which technique(s) apply to the target sentence.

Surrounding Context:
{context_block}

Candidate Techniques:
{candidates_formatted}

Provide your answer as a list of technique identifiers and names in the format:
Txxxx : Name
If no techniques apply, output 'None'.

Target Sentence: {sentence}"""

# Reasoning Prompt (DeepSeek-R1 style)
REASONING_PROMPT = """You are a cybersecurity expert. Analyze the following Cyber Threat Intelligence (CTI) report sentence and identify the relevant MITRE ATT&CK techniques from the provided candidates.
First, reason step-by-step about what adversary behavior is occurring.
Then, output your final technique selection formatted strictly as:
Txxxx : Name
If no techniques apply, output 'None'.

Candidate Techniques:
{candidates_formatted}

Sentence: {sentence}"""

# Appendix F: Few-Shot Chat Examples (3 examples: single-label, multi-label, none)
FEW_SHOT_CHAT_EXAMPLES: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": "The adversary executed command-line scripts using cmd.exe to launch secondary processes.",
    },
    {
        "role": "assistant",
        "content": "T1059.003 : Windows Command Shell",
    },
    {
        "role": "user",
        "content": "The attackers used PsExec to execute malicious binaries on remote workstations and established an RDP connection.",
    },
    {
        "role": "assistant",
        "content": "T1569.002 : Service Execution\nT1021.001 : Remote Desktop Protocol",
    },
    {
        "role": "user",
        "content": "Security teams should ensure that multi-factor authentication is enforced across all enterprise VPN gateways.",
    },
    {
        "role": "assistant",
        "content": "None",
    },
]


def format_candidates(
    candidates: List[Union[RetrievedCandidate, Dict[str, Any]]],
    include_description: bool = True,
    desc_mode: str = "first_paragraph",  # 'first_paragraph', 'full', 'summarized'
    max_desc_len: int = 300,
) -> str:
    """Formats retrieved candidates into the paper's candidate text block."""
    if not candidates:
        return "None provided."

    lines = []
    for idx, c in enumerate(candidates, start=1):
        if isinstance(c, RetrievedCandidate):
            tid = c.technique_id
            name = c.technique_name
            desc = c.description
        else:
            tid = c.get("technique_id", "")
            name = c.get("technique_name", "")
            desc = c.get(f"{desc_mode}_description", c.get("description", ""))

        if include_description:
            clean_desc = desc.strip()
            if len(clean_desc) > max_desc_len:
                clean_desc = clean_desc[:max_desc_len].rsplit(" ", 1)[0] + "..."
            lines.append(f"{idx}. {tid} : {name}\n   Description: {clean_desc}")
        else:
            lines.append(f"{idx}. {tid} : {name}")

    return "\n".join(lines)


def build_reproduction_prompt(
    experiment_type: str,
    sentence: str,
    candidates: Optional[List[Union[RetrievedCandidate, Dict[str, Any]]]] = None,
    prev_sentence: Optional[str] = None,
    next_sentence: Optional[str] = None,
    prev_2: Optional[str] = None,
    next_2: Optional[str] = None,
    k: int = 10,
    include_description: bool = True,
    desc_mode: str = "first_paragraph",
) -> Union[str, List[Dict[str, str]]]:
    """
    Builds the exact prompt or chat message history for the requested paper experiment.
    """
    cands_str = format_candidates(
        candidates or [],
        include_description=include_description,
        desc_mode=desc_mode,
    )

    if experiment_type == "base_llm":
        return APPENDIX_B_BASE_LLM.format(sentence=sentence)

    elif experiment_type == "base_rag":
        return APPENDIX_C_BASE_RAG.format(
            candidates_formatted=cands_str,
            sentence=sentence,
        )

    elif experiment_type == "optimized_rag":
        return APPENDIX_D_OPTIMIZED_RAG.format(
            k=k,
            candidates_formatted=cands_str,
            sentence=sentence,
        )

    elif experiment_type == "context_1":
        context_parts = []
        if prev_sentence:
            context_parts.append(f"[Previous]: {prev_sentence}")
        context_parts.append(f"[Target]: {sentence}")
        if next_sentence:
            context_parts.append(f"[Following]: {next_sentence}")
        context_block = "\n".join(context_parts)

        return APPENDIX_E_CONTEXT.format(
            context_block=context_block,
            candidates_formatted=cands_str,
            sentence=sentence,
        )

    elif experiment_type == "context_2":
        context_parts = []
        if prev_2:
            context_parts.append(f"[Previous 2]: {prev_2}")
        if prev_sentence:
            context_parts.append(f"[Previous 1]: {prev_sentence}")
        context_parts.append(f"[Target]: {sentence}")
        if next_sentence:
            context_parts.append(f"[Following 1]: {next_sentence}")
        if next_2:
            context_parts.append(f"[Following 2]: {next_2}")
        context_block = "\n".join(context_parts)

        return APPENDIX_E_CONTEXT.format(
            context_block=context_block,
            candidates_formatted=cands_str,
            sentence=sentence,
        )

    elif experiment_type == "few_shot":
        # Multi-turn chat message sequence
        system_msg = {
            "role": "system",
            "content": "You are a cybersecurity expert. Given a CTI sentence and candidate MITRE ATT&CK techniques, output relevant technique IDs and names in format 'Txxxx : Name' or 'None'.",
        }
        user_prompt = f"Candidate Techniques:\n{cands_str}\n\nSentence: {sentence}"
        messages = [system_msg] + FEW_SHOT_CHAT_EXAMPLES + [{"role": "user", "content": user_prompt}]
        return messages

    elif experiment_type == "reasoning":
        return REASONING_PROMPT.format(
            candidates_formatted=cands_str,
            sentence=sentence,
        )

    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")
