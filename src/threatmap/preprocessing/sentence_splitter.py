"""
Cybersecurity-aware sentence splitter.
Preserves technical tokens such as IPs, URLs, technique IDs (T1059.001),
executables (cmd.exe, powershell.exe), CVEs, and registry paths.
"""

import re
from typing import List

# Known abbreviations that should not trigger a sentence split
ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "al.", "approx.", "fig.", "no.", "dr.", "mr.", "ms.",
}

# Regex to detect protected technical patterns that contain periods
PROTECTED_PATTERNS = [
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",                      # IPv4 addresses
    r"\bhttps?://[^\s]+",                                            # URLs
    r"\bT\d{4}\.\d{3}\b",                                            # MITRE Subtechnique IDs (T1059.001)
    r"\bCVE-\d{4}-\d{4,7}\b",                                        # CVE IDs
    r"\b[A-Za-z0-9_\-]+\.(?:exe|dll|ps1|bat|vbs|sys|bin|sh|py|docx|xlsx|pdf|zip|rar|7z)\b",  # File extensions
    r"\bv\d+(?:\.\d+)+\b",                                           # Version strings like v1.3.2
]

COMPILED_PROTECTED = re.compile("|".join(f"({p})" for p in PROTECTED_PATTERNS), re.IGNORECASE)


def normalize_text_preservative(text: str) -> str:
    """Normalizes whitespace and standardizes quotes without stripping security tokens."""
    if not text:
        return ""
    # Standardize unicode quotes and dashes
    cleaned = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = cleaned.replace("—", " - ").replace("–", " - ")
    # Normalize excessive carriage returns and non-breaking spaces
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    return cleaned.strip()


def split_into_sentences(text: str, min_length: int = 5) -> List[str]:
    """
    Splits text into sentences while strictly preserving cybersecurity tokens.
    """
    normalized = normalize_text_preservative(text)
    if not normalized:
        return []

    # Replace periods in protected tokens with a special placeholder token
    placeholder_map = {}
    counter = 0

    def mask_match(match):
        nonlocal counter
        matched_str = match.group(0)
        trailing_punct = ""
        # If the matched token ends with trailing punctuation, strip it so it serves as punctuation
        while matched_str and matched_str[-1] in ".,!?:;\"'":
            trailing_punct = matched_str[-1] + trailing_punct
            matched_str = matched_str[:-1]

        if "." in matched_str:
            placeholder = f"__TOKEN_PERIOD_{counter}__"
            placeholder_map[placeholder] = matched_str
            counter += 1
            return placeholder + trailing_punct
        return matched_str + trailing_punct

    masked_text = COMPILED_PROTECTED.sub(mask_match, normalized)

    # Also protect known abbreviations
    for abbr in ABBREVIATIONS:
        if abbr in masked_text.lower():
            # case insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            placeholder = f"__ABBR_PERIOD_{counter}__"
            # find actual string matching
            match = pattern.search(masked_text)
            if match:
                placeholder_map[placeholder] = match.group(0)
                masked_text = pattern.sub(placeholder, masked_text)
                counter += 1

    # Split on sentence boundaries:
    # 1. Newline splits (paragraphs)
    # 2. Punctuation (. ! ?) followed by whitespace and uppercase letter or EOF
    sentence_boundary_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?<=\n)\s*")
    raw_splits = sentence_boundary_pattern.split(masked_text)

    sentences: List[str] = []
    for chunk in raw_splits:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Restore placeholders
        restored = chunk
        for placeholder, original in placeholder_map.items():
            restored = restored.replace(placeholder, original)

        # Collapse internal multiple whitespaces
        restored = " ".join(restored.split())

        if len(restored) >= min_length:
            sentences.append(restored)

    return sentences
