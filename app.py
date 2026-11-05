"""
ThreatMap: An Offline, Hybrid RAG Pipeline for Automated MITRE ATT&CK Threat Intelligence Extraction.
Interactive Cybersecurity Threat Intelligence Dashboard.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from threatmap.config import load_config
from threatmap.classification.pipeline import ThreatMapPipeline
from threatmap.data.attack_loader import load_attack_data, AttackKnowledgeBase
from threatmap.data.tram_loader import load_benchmark_dataset
from threatmap.evaluation.benchmarks import run_ablation_benchmarks
from threatmap.evaluation.evaluator import PipelineEvaluator
from threatmap.export.csv_export import export_to_csv
from threatmap.export.json_export import export_to_json, export_to_jsonl
from threatmap.retrieval.hybrid import HybridRetriever
from threatmap.retrieval.index_manager import IndexManager
from threatmap.retrieval.lexical import BM25Retriever
from threatmap.retrieval.semantic import SemanticRetriever
from threatmap.schemas import AttackTechnique

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ThreatMap | MITRE ATT&CK RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-predicted {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }
    .badge-abstained {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    .evidence-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 10px 14px;
        margin-top: 6px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: #f1f5f9;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# RAG-ATT&CK constants (must match run_llm_sample.py exactly)
# ---------------------------------------------------------
RAG_MODEL = "llama3.1:8b-instruct-q4_0"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
RAG_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 64, "num_ctx": 4096}
RAG_K = 10  # candidates shown to LLM
KB_50_PATH = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"
KB_FULL_PATH = BASE_DIR / "data" / "processed" / "mitre_attack_enterprise.json"

CODE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b")

# ---------------------------------------------------------
# KB & Pipeline helpers
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading 50-technique knowledge base...")
def load_kb_50() -> List[Dict]:
    """Load the 50 paper techniques as plain dicts (same format as run_llm_sample.py)."""
    with open(KB_50_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["techniques"]


@st.cache_resource(show_spinner="Initializing ThreatMap pipeline (50 techniques)...")
def get_pipeline_50():
    config = load_config(base_dir=BASE_DIR)
    # Override KB path to 50-technique set
    config.paths.processed_attack_file = str(
        KB_50_PATH.relative_to(BASE_DIR)
    )
    kb = load_attack_data(
        processed_path=BASE_DIR / config.paths.processed_attack_file,
        raw_path=BASE_DIR / config.paths.raw_attack_file,
        fallback_to_fixture=True,
    )
    # Build a fresh retriever over the 50 techniques only (never touches artifacts/)
    sem = SemanticRetriever(
        model_name=config.retrieval.embedding_model,
        device=config.retrieval.device,
        batch_size=config.retrieval.batch_size,
    )
    sem.index_techniques(kb.get_all_active_techniques())
    bm25 = BM25Retriever(kb.get_all_active_techniques())
    retriever = HybridRetriever(
        bm25_retriever=bm25,
        semantic_retriever=sem,
        default_mode=config.retrieval.hybrid_mode,
        rrf_k=config.retrieval.rrf_k,
    )
    return ThreatMapPipeline(config=config, kb=kb, retriever=retriever, base_dir=BASE_DIR)


@st.cache_resource(show_spinner="Initializing ThreatMap pipeline (full ATT&CK)...")
def get_pipeline_full():
    config = load_config(base_dir=BASE_DIR)
    return ThreatMapPipeline(config=config, base_dir=BASE_DIR)


# ---------------------------------------------------------
# RAG-ATT&CK LLM helpers (identical prompt to run_llm_sample.py #22)
# ---------------------------------------------------------

def _format_candidates(candidates: List[Dict]) -> str:
    return "\n".join(
        f"[technique: {t['technique_id']} - {t['technique_name']}]"
        for t in candidates
    )


def build_rag_messages(sentence: str, candidates: List[Dict]) -> List[Dict]:
    """Exact Appendix D.1 prompt used in experiment #22 (BaseRAG)."""
    cands_str = _format_candidates(candidates)
    system_content = (
        "Task Instructions:\n"
        "Label the sentences provided by the user with a corresponding MITRE ATT&CK Technique. "
        'If no applicable technique exists, please respond with the string "No Mapping."\n\n'
        "Guidelines:\n"
        "- You may assign multiple MITRE ATT&CK techniques to a sentence if relevant.\n"
        "- Limit your mappings to the following MITRE ATT&CK techniques:\n"
        f"{cands_str}\n\n"
        "Now create a list of applicable MITRE ATT&CK techniques to the sentence inputted by the user.\n\n"
        "Output Format:\n"
        'Your response should be formatted as follows: "Txxxx : Name"'
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": sentence},
    ]


def call_ollama_rag(messages: List[Dict]) -> Tuple[str, str]:
    """
    Call Ollama with RAG-ATT&CK settings.
    Returns (raw_reply, error_message).
    error_message is "" on success.
    """
    payload = {
        "model": RAG_MODEL,
        "messages": messages,
        "stream": False,
        "options": RAG_OPTIONS,
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        return "", (
            f"Ollama is unreachable at {OLLAMA_CHAT_URL}. "
            "Start Ollama with: `ollama serve` and ensure the model is pulled: "
            f"`ollama pull {RAG_MODEL}`"
        )
    except requests.exceptions.Timeout:
        return "", f"Ollama timed out after 120 s (model: {RAG_MODEL})."

    if resp.status_code == 404:
        return "", (
            f"Model not found: {RAG_MODEL}. "
            f"Pull it with: `ollama pull {RAG_MODEL}`"
        )
    if resp.status_code != 200:
        return "", f"Ollama HTTP {resp.status_code}: {resp.text[:200]}"

    content = resp.json().get("message", {}).get("content", "")
    return content, ""


def parse_rag_reply(raw: str, candidate_ids: set) -> List[str]:
    """
    Parse technique IDs from raw LLM reply using regex \\b(T\\d{4}(\\.\\d{3})?)\\b.
    Returns list of parsed codes (may be empty -> NO_MAPPING).
    Does NOT filter to candidates — reports whatever the model returned.
    """
    if not raw:
        return []
    no_mapping_patterns = [
        "no mapping", "no applicable", "no technique", "no mitre"
    ]
    lower = raw.lower()
    if any(p in lower for p in no_mapping_patterns) and not CODE_RE.search(raw):
        return []
    return list(dict.fromkeys(CODE_RE.findall(raw)))  # unique, order-preserved


# ---------------------------------------------------------
# Sentence splitter (simple, no NLTK needed)
# ---------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split text into non-empty stripped lines / sentences for display."""
    # Split on newlines first to preserve paragraph structure
    lines = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        # Further split on sentence boundaries only if line is long
        if len(block) > 300:
            # Basic split on ". " / "! " / "? "
            parts = re.split(r"(?<=[.!?])\s+", block)
            lines.extend(p.strip() for p in parts if p.strip())
        else:
            lines.append(block)
    return lines


# ---------------------------------------------------------
# RAG-ATT&CK analysis (new, sentence-level with LLM)
# ---------------------------------------------------------

def rag_attack_analyze(
    sentences: List[str],
    pipeline: ThreatMapPipeline,
    techniques_50: List[Dict],
    retrieval_mode: str = "hybrid_rrf",
) -> List[Dict]:
    """
    For each sentence:
      1. Retrieve top-10 candidates via pipeline retriever
      2. Build Appendix D.1 prompt
      3. Call Ollama (RAG_MODEL)
      4. Parse codes; NO_MAPPING if none
    Returns list of result dicts.
    """
    results = []
    for sent in sentences:
        # --- retrieval ---
        raw_cands = pipeline.retriever.retrieve(
            query=sent,
            mode=retrieval_mode,
            top_k=RAG_K,
        )
        # Map retrieved candidates to their plain-dict form for the prompt
        cand_ids = [c.technique_id for c in raw_cands]
        cand_dicts = []
        for cid in cand_ids:
            match = next(
                (t for t in techniques_50 if t["technique_id"] == cid),
                None,
            )
            if match:
                cand_dicts.append(match)
            else:
                # fallback: use retrieved name
                rc = next((c for c in raw_cands if c.technique_id == cid), None)
                if rc:
                    cand_dicts.append(
                        {"technique_id": rc.technique_id, "technique_name": rc.technique_name}
                    )

        messages = build_rag_messages(sent, cand_dicts)
        raw_reply, error = call_ollama_rag(messages)

        if error:
            results.append(
                {
                    "sentence": sent,
                    "candidates": raw_cands,
                    "raw_reply": None,
                    "labels": [],
                    "error": error,
                }
            )
            continue

        parsed = parse_rag_reply(raw_reply, set(cand_ids))
        results.append(
            {
                "sentence": sent,
                "candidates": raw_cands,
                "raw_reply": raw_reply,
                "labels": parsed,  # empty list = NO_MAPPING
                "error": "",
            }
        )
    return results


# ---------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------

def ollama_is_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------
# Sidebar — KB toggle + navigation
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛡️ ThreatMap")
    st.caption("Offline Hybrid RAG for MITRE ATT&CK Threat Intelligence")

    page = st.radio(
        "Navigation",
        [
            "Overview & Architecture",
            "Analyze Threat Report",
            "Results & Evidence",
            "MITRE ATT&CK Explorer",
            "Evaluation Lab",
            "Settings & Offline Status",
        ],
    )

    st.divider()
    st.markdown("### Knowledge Base")
    use_full_kb = st.toggle(
        "Use full ATT&CK (~700 techniques)",
        value=False,
        help=(
            "OFF (default): restrict to the 50 paper techniques from "
            "mitre_attack_50_techniques.json, matching the RAG-ATT&CK paper scope.\n\n"
            "ON: search all active ATT&CK enterprise techniques (~700). "
            "New IDs outside the paper set will appear."
        ),
    )

    if use_full_kb:
        pipeline = get_pipeline_full()
        kb_label = f"Full ATT&CK ({len(pipeline.kb)} techniques)"
    else:
        pipeline = get_pipeline_50()
        kb_label = "Paper 50 techniques"

    st.markdown("### System Health")
    st.write(f"**Knowledge Base:** `{kb_label}`")
    st.write(f"**Embedding Model:** `all-MiniLM-L6-v2`")
    st.write(f"**BM25 Index:** `Active`")

    ollama_ok = ollama_is_available()
    if ollama_ok:
        st.success("Ollama: Connected")
    else:
        st.warning("Ollama: Offline")

    st.caption("ThreatMap v0.1.0 | Academic Research Project")

# Load 50-technique plain dicts (used for prompt building)
techniques_50 = load_kb_50()

# ---------------------------------------------------------
# Page 1: Overview & Architecture
# ---------------------------------------------------------
if page == "Overview & Architecture":
    st.header("ThreatMap: Offline Hybrid RAG for Threat Intelligence")
    st.markdown(
        """
        **ThreatMap** is a research-oriented NLP system that extracts adversary behaviors from unstructured cybersecurity reports
        and maps them to structured **MITRE ATT&CK** techniques using a **Hybrid Retrieval-Augmented Generation (RAG)** architecture.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">ATT&CK Knowledge Base</div>
                <div class="metric-value">{len(pipeline.kb)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Retrieval Mode</div>
                <div class="metric-value" style="font-size:1.4rem;">Hybrid RRF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Privacy Level</div>
                <div class="metric-value" style="font-size:1.4rem; color:#16a34a;">100% Offline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Local LLM Status</div>
                <div class="metric-value" style="font-size:1.3rem;">{'Online' if ollama_ok else 'Offline'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Key Architecture & Capabilities")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            * **Lexical Retrieval (BM25 Okapi)**: Indexes exact command names, PowerShell cmdlets, registry keys, and ATT&CK IDs.
            * **Dense Semantic Retrieval**: Employs Sentence Transformers to capture conceptual adversary descriptions via cosine similarity.
            * **Reciprocal Rank Fusion (RRF)**: Combines lexical and semantic ranks to maximize candidate recall without arbitrary score weighting.
            * **Cyber-Aware Segmentation**: Sentence splitter prevents boundary collapse on IP addresses, file extensions (`cmd.exe`), and CVEs.
            """
        )
    with col_b:
        st.markdown(
            """
            * **LLM Selection (RAG-ATT&CK)**: Uses the exact Appendix D.1 prompt from the paper — 10 retrieved candidates, `llama3.1:8b-instruct-q4_0`, temperature=0, seed=42.
            * **NO_MAPPING Support**: Model can abstain. Non-adversarial sentences get `NO_MAPPING` rather than a forced label.
            * **50-Technique Default**: Knowledge base restricted to the 50 paper techniques by default; full ATT&CK available via sidebar toggle.
            * **Local-First & Zero-Cloud**: Operates completely on-premise without transmitting sensitive threat data to third-party APIs.
            """
        )

    st.subheader("Research Grounding")
    st.info(
        """
        **Methodological Foundation**: Reproduces the RAG pipeline from *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence
        to MITRE ATT&CK Techniques* (Stijn Schuurman, University of Twente, 2024).
        Default KB = 50 paper techniques. LLM = `llama3.1:8b-instruct-q4_0` via local Ollama.
        Prompt = verbatim Appendix D.1 (BaseRAG / experiment #22).
        """
    )


# ---------------------------------------------------------
# Page 2: Analyze Threat Report
# ---------------------------------------------------------
elif page == "Analyze Threat Report":
    st.header("Analyze Cyber Threat Intelligence Report")
    st.markdown(
        "Input an unstructured threat report to extract MITRE ATT&CK techniques using the RAG-ATT&CK pipeline."
    )

    # Sample text
    SAMPLE_REPORT_PATH = BASE_DIR / "data" / "samples" / "sample_threat_report.txt"
    DEFAULT_SAMPLE_TEXT = (
        SAMPLE_REPORT_PATH.read_text(encoding="utf-8")
        if SAMPLE_REPORT_PATH.exists()
        else (
            "Introduction\n"
            "This report describes a targeted spear-phishing campaign.\n"
            "The attacker sent spear-phishing emails with malicious Word attachments containing macro code.\n"
            "When the victim opened the document and enabled macros, a PowerShell script was executed.\n"
            "The PowerShell script downloaded a Cobalt Strike beacon from a remote server.\n"
            "Credentials were dumped from LSASS memory using Mimikatz.\n"
            "The attacker moved laterally via Remote Desktop Protocol using stolen credentials.\n"
            "The malware decoded its configuration using base64 decoding.\n"
            "The attacker exploited a vulnerable public-facing web server to gain initial access.\n"
            "Recommendations: enforce multi-factor authentication on all remote access points."
        )
    )

    col_input_opt, col_file = st.columns([2, 1])
    with col_input_opt:
        if st.button("📋 Load Sample Report"):
            st.session_state["report_input"] = DEFAULT_SAMPLE_TEXT
    with col_file:
        uploaded_file = st.file_uploader("Upload TXT / PDF report", type=["txt", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                try:
                    from threatmap.preprocessing.document_parser import extract_text_from_pdf
                    st.session_state["report_input"] = extract_text_from_pdf(uploaded_file.getvalue())
                    st.success(f"Extracted PDF: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
            else:
                st.session_state["report_input"] = uploaded_file.getvalue().decode("utf-8", errors="replace")
                st.success(f"Loaded: {uploaded_file.name}")

    report_text = st.text_area(
        "Threat Report Content",
        value=st.session_state.get("report_input", DEFAULT_SAMPLE_TEXT),
        height=240,
        placeholder="Paste plain text threat report here...",
    )

    # Mode selector
    st.subheader("Analysis Mode")
    mode_col, param_col = st.columns([1, 1])
    with mode_col:
        analysis_mode = st.radio(
            "Select mode",
            [
                "LLM selection (RAG-ATT&CK)",
                "retrieval only (forces a label on every sentence)",
            ],
            index=0,
            help=(
                "**LLM selection**: retrieve 10 candidates, ask llama3.1:8b-instruct-q4_0 to pick "
                "(Appendix D.1 prompt). Outputs NO_MAPPING for non-attack sentences. Requires Ollama.\n\n"
                "**Retrieval only**: take the top-1 retrieved technique for every sentence. "
                "Every sentence gets a label — no abstention."
            ),
        )
    with param_col:
        retrieval_mode = st.selectbox(
            "Retrieval Strategy",
            ["hybrid_rrf", "hybrid_weighted", "semantic", "lexical"],
            index=0,
        )
        if analysis_mode.startswith("retrieval"):
            top_k_ret = st.slider("Top-K (retrieval only)", 1, 10, 1)
        else:
            st.info(f"Top-K fixed at {RAG_K} for LLM selection mode.")
            top_k_ret = RAG_K

    if st.button("🚀 Run ThreatMap Engine", type="primary", use_container_width=True):
        if not report_text.strip():
            st.warning("Please provide a threat report before running analysis.")
        elif analysis_mode.startswith("LLM") and not ollama_ok:
            st.error(
                f"**Ollama is unreachable.** Cannot run LLM selection mode.\n\n"
                f"To fix:\n"
                f"1. Open a terminal and run: `ollama serve`\n"
                f"2. Pull the required model: `ollama pull {RAG_MODEL}`\n"
                f"3. Reload this page and try again.\n\n"
                f"Or switch to **retrieval only** mode (no Ollama required)."
            )
        else:
            sentences = split_sentences(report_text)
            st.session_state["sentences"] = sentences

            if analysis_mode.startswith("LLM"):
                # --- RAG-ATT&CK LLM mode ---
                with st.spinner(
                    f"Running RAG-ATT&CK pipeline ({len(sentences)} sentences, "
                    f"model={RAG_MODEL})..."
                ):
                    rag_results = rag_attack_analyze(
                        sentences=sentences,
                        pipeline=pipeline,
                        techniques_50=techniques_50,
                        retrieval_mode=retrieval_mode,
                    )
                    # Check for Ollama errors
                    errors = [r for r in rag_results if r["error"]]
                    if errors:
                        st.error(
                            f"**Ollama error on sentence 1:** {errors[0]['error']}\n\n"
                            "Analysis stopped. Fix Ollama and retry."
                        )
                    else:
                        st.session_state["rag_results"] = rag_results
                        st.session_state["analysis_mode_used"] = "llm"
                        st.success(
                            f"Done! {len(sentences)} sentences processed. "
                            f"{sum(1 for r in rag_results if r['labels'])} mapped, "
                            f"{sum(1 for r in rag_results if not r['labels'])} NO_MAPPING."
                        )
            else:
                # --- Retrieval-only mode ---
                with st.spinner(f"Running retrieval-only pipeline ({len(sentences)} sentences)..."):
                    analysis_result = pipeline.analyze_report(
                        content=report_text,
                        filename="Uploaded_Report.txt",
                        retrieval_mode=retrieval_mode,
                        top_k=top_k_ret,
                        use_llm=False,
                        context_strategy="target_with_candidates",
                    )
                    st.session_state["analysis_result"] = analysis_result
                    st.session_state["analysis_mode_used"] = "retrieval"
                    st.success(
                        f"Analysis Complete! Processed {analysis_result.total_sentences} sentences. "
                        f"Found {len(analysis_result.unique_techniques)} unique ATT&CK techniques."
                    )

    if "rag_results" in st.session_state and st.session_state.get("analysis_mode_used") == "llm":
        st.info("LLM results ready — see the **Results & Evidence** tab for the full table.")
    elif "analysis_result" in st.session_state and st.session_state.get("analysis_mode_used") == "retrieval":
        st.info("Retrieval results ready — see the **Results & Evidence** tab.")


# ---------------------------------------------------------
# Page 3: Results & Evidence
# ---------------------------------------------------------
elif page == "Results & Evidence":
    st.header("Threat Intelligence Extraction Results")

    mode_used = st.session_state.get("analysis_mode_used", None)

    # ---- LLM (RAG-ATT&CK) results ----
    if mode_used == "llm" and "rag_results" in st.session_state:
        rag_results = st.session_state["rag_results"]

        # Summary metrics
        n_total = len(rag_results)
        n_mapped = sum(1 for r in rag_results if r["labels"])
        n_no_map = n_total - n_mapped

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Sentences", n_total)
        with m2:
            st.metric("Mapped (technique assigned)", n_mapped)
        with m3:
            st.metric("NO_MAPPING", n_no_map)

        st.divider()
        st.subheader("Sentence-Level Results (RAG-ATT&CK / LLM selection)")
        st.caption(
            f"Model: `{RAG_MODEL}` | temperature=0, seed=42 | "
            f"Prompt: Appendix D.1 (BaseRAG/#22) | Candidates: top-{RAG_K}"
        )

        for i, r in enumerate(rag_results, 1):
            labels_str = ", ".join(r["labels"]) if r["labels"] else "**NO_MAPPING**"
            with st.expander(
                f"[{i}] {r['sentence'][:90]}{'...' if len(r['sentence']) > 90 else ''}  →  {labels_str}",
                expanded=False,
            ):
                st.markdown(f"**Sentence:** {r['sentence']}")
                st.markdown(f"**Final labels:** `{labels_str}`")

                # Candidates table
                st.markdown(f"**Top-{RAG_K} retrieved candidates:**")
                cand_rows = [
                    {
                        "Rank": c.fused_rank,
                        "ID": c.technique_id,
                        "Name": c.technique_name,
                        "Sem Score": c.semantic_score,
                        "Method": c.retrieval_method,
                    }
                    for c in r["candidates"]
                ]
                st.dataframe(pd.DataFrame(cand_rows), use_container_width=True, height=250)

                # Raw LLM reply
                st.markdown("**Model's raw reply:**")
                st.code(r["raw_reply"] or "(no reply)", language=None)

        # Flat summary table
        st.divider()
        st.subheader("Summary Table")
        rows = []
        for r in rag_results:
            rows.append(
                {
                    "Sentence": r["sentence"][:100],
                    "Labels": ", ".join(r["labels"]) if r["labels"] else "NO_MAPPING",
                    "Candidates": " | ".join(c.technique_id for c in r["candidates"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ---- Retrieval-only results ----
    elif mode_used == "retrieval" and "analysis_result" in st.session_state:
        res = st.session_state["analysis_result"]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Total Sentences</div>
                    <div class="metric-value">{res.total_sentences}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Sentences with Techniques</div>
                    <div class="metric-value" style="color:#0284c7;">{res.sentences_with_predictions}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Abstained</div>
                    <div class="metric-value" style="color:#64748b;">{res.abstention_count}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Unique Techniques</div>
                    <div class="metric-value" style="color:#16a34a;">{len(res.unique_techniques)}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")

        if res.tactic_distribution:
            col_chart, col_meta = st.columns([2, 1])
            with col_chart:
                st.subheader("Tactics Distribution")
                df_tactics = pd.DataFrame(
                    list(res.tactic_distribution.items()),
                    columns=["Tactic", "Count"],
                ).sort_values(by="Count", ascending=False)
                fig = px.bar(
                    df_tactics,
                    x="Tactic",
                    y="Count",
                    color="Count",
                    color_continuous_scale="Blues",
                    text="Count",
                )
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col_meta:
                st.subheader("Audit Metadata")
                meta = res.pipeline_metadata
                st.write(f"**Retrieval Method:** `{meta.get('retrieval_mode')}`")
                st.write(f"**Top-K Candidates:** `{meta.get('top_k')}`")
                st.write(f"**Mode:** `retrieval only`")
                st.write(f"**Document ID:** `{res.document_id}`")
                st.write(f"**Generated:** `{res.timestamp[:19]}`")

        st.subheader("Sentence-Level Evidence & Technique Mappings")
        table_rows = []
        for s in res.sentence_results:
            if s.abstained or not s.predictions:
                table_rows.append(
                    {
                        "ID": s.sentence_id,
                        "Status": "ABSTAINED",
                        "Technique ID": "-",
                        "Technique Name": "-",
                        "Tactic": "-",
                        "Score": 0.0,
                        "Sentence Text": s.original_sentence,
                        "Explanation / Reason": s.abstention_reason or "Non-adversarial / Low confidence",
                    }
                )
            else:
                for p in s.predictions:
                    table_rows.append(
                        {
                            "ID": s.sentence_id,
                            "Status": "PREDICTED",
                            "Technique ID": p.technique_id,
                            "Technique Name": p.technique_name,
                            "Tactic": p.tactic or "Unknown",
                            "Score": p.retrieval_score,
                            "Sentence Text": s.original_sentence,
                            "Explanation / Reason": p.explanation,
                        }
                    )

        df_display = pd.DataFrame(table_rows)
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            status_filter = st.multiselect(
                "Filter by Status", ["PREDICTED", "ABSTAINED"], default=["PREDICTED", "ABSTAINED"]
            )
        with f_col2:
            search_query = st.text_input("Search Sentence Text or Technique ID", "")

        filtered_df = df_display[df_display["Status"].isin(status_filter)]
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["Sentence Text"].str.lower().str.contains(q)
                | filtered_df["Technique ID"].str.lower().str.contains(q)
                | filtered_df["Technique Name"].str.lower().str.contains(q)
            ]
        st.dataframe(filtered_df, use_container_width=True, height=350)

        # Sentence inspector
        st.subheader("🔍 Granular Sentence & Evidence Inspector")
        sentence_ids = [s.sentence_id for s in res.sentence_results]
        selected_sid = st.selectbox("Select a sentence to inspect", sentence_ids)
        target_sent = next((s for s in res.sentence_results if s.sentence_id == selected_sid), None)
        if target_sent:
            ins_col1, ins_col2 = st.columns([1, 1])
            with ins_col1:
                st.markdown(f"**Target Sentence ({target_sent.sentence_id}):**")
                st.info(target_sent.original_sentence)
                if target_sent.context_text:
                    with st.expander("Surrounding Sentence Context"):
                        st.code(target_sent.context_text)
                if target_sent.abstained:
                    st.warning(f"**Status: ABSTAINED** — {target_sent.abstention_reason}")
                else:
                    st.success(f"**Status: PREDICTED ({len(target_sent.predictions)} technique(s))**")
                    for p in target_sent.predictions:
                        st.markdown(f"### `{p.technique_id}`: {p.technique_name}")
                        st.markdown(f"**Tactic:** `{p.tactic}` | **Score:** `{p.retrieval_score}`")
                        st.markdown(
                            f"<div class='evidence-box'><strong>Evidence Quote:</strong> \"{p.evidence_quote}\"</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Explanation:** {p.explanation}")
            with ins_col2:
                st.markdown("**Top Retrieved Candidates:**")
                if target_sent.alternative_candidates:
                    cand_data = [
                        {
                            "Rank": c.fused_rank,
                            "ID": c.technique_id,
                            "Name": c.technique_name,
                            "Lexical Rank": c.lexical_rank,
                            "Semantic Rank": c.semantic_rank,
                            "Fused Score": c.fused_score,
                        }
                        for c in target_sent.alternative_candidates
                    ]
                    st.dataframe(pd.DataFrame(cand_data), use_container_width=True)
                else:
                    st.write("No candidate techniques retrieved.")

        # Export
        st.divider()
        st.subheader("Export Results")
        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            csv_str = export_to_csv(res)
            st.download_button(
                "📥 Download CSV",
                data=csv_str,
                file_name=f"threatmap_{res.document_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp2:
            json_str = export_to_json(res)
            st.download_button(
                "📥 Download JSON",
                data=json_str,
                file_name=f"threatmap_{res.document_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp3:
            jsonl_str = export_to_jsonl(res)
            st.download_button(
                "📥 Download JSONL",
                data=jsonl_str,
                file_name=f"threatmap_{res.document_id}.jsonl",
                mime="application/x-ndjson",
                use_container_width=True,
            )

    else:
        st.warning(
            "No analysis results found. Go to **Analyze Threat Report** and run the pipeline first."
        )


# ---------------------------------------------------------
# Page 4: MITRE ATT&CK Explorer
# ---------------------------------------------------------
elif page == "MITRE ATT&CK Explorer":
    st.header("MITRE ATT&CK Knowledge Base Explorer")
    st.markdown(f"Explore the local knowledge base: **{len(pipeline.kb)}** active techniques.")

    search_query = st.text_input("Search by Technique Name, ID (e.g. T1059.001), or keyword", "")
    all_tactics = sorted(
        list(
            {
                tactic
                for t in pipeline.kb.get_all_active_techniques()
                for tactic in t.tactic_names
            }
        )
    )
    selected_tactic = st.selectbox("Filter by Tactic", ["All Tactics"] + all_tactics)

    techniques = pipeline.kb.get_all_active_techniques()
    if selected_tactic != "All Tactics":
        techniques = [t for t in techniques if selected_tactic in t.tactic_names]
    if search_query:
        q = search_query.lower()
        techniques = [
            t
            for t in techniques
            if q in t.technique_id.lower()
            or q in t.technique_name.lower()
            or q in t.description.lower()
        ]

    st.write(f"Showing **{len(techniques)}** matching techniques:")
    for tech in techniques[:25]:
        with st.expander(f"`{tech.technique_id}` — {tech.technique_name}"):
            st.write(f"**Tactics:** {', '.join(tech.tactic_names) if tech.tactic_names else 'General'}")
            st.write(f"**Platforms:** {', '.join(tech.platforms) if tech.platforms else 'All'}")
            if tech.is_subtechnique:
                st.write(f"**Sub-technique of:** `{tech.subtechnique_of}`")
            st.markdown(f"**Description:**\n{tech.description}")
            if tech.detection:
                st.markdown(f"**Detection Guidance:**\n{tech.detection}")
            if tech.source_url:
                st.markdown(f"[Official MITRE ATT&CK Link]({tech.source_url})")


# ---------------------------------------------------------
# Page 5: Evaluation Lab
# ---------------------------------------------------------
elif page == "Evaluation Lab":
    st.header("ThreatMap Quantitative Evaluation Lab")
    st.markdown(
        """
        Evaluate retrieval and classification accuracy using multilabel metrics (Micro/Macro F1, Recall@k, MRR, Coverage).
        """
    )

    bench_path = BASE_DIR / "data" / "evaluation" / "tram_sample_benchmark.json"
    benchmark_examples = load_benchmark_dataset(
        dataset_path=bench_path,
        valid_technique_ids=pipeline.kb.get_technique_ids(),
    )
    st.write(f"**Benchmark Dataset:** `TRAM Sample Benchmark` ({len(benchmark_examples)} annotated sentences)")

    eval_tab1, eval_tab2 = st.tabs(["Single Evaluation Run", "Full Ablation Experiments (A - H)"])

    with eval_tab1:
        c1, c2 = st.columns(2)
        with c1:
            eval_mode = st.selectbox("Retrieval Method", ["hybrid_rrf", "hybrid_weighted", "semantic", "lexical"], index=0)
            eval_top_k = st.slider("Evaluation Top-K", 1, 10, 5)
        with c2:
            eval_llm = st.checkbox(
                "Validate with Local LLM (if available)", value=False, disabled=not ollama_ok
            )

        if st.button("▶️ Execute Benchmark Evaluation", type="primary"):
            with st.spinner("Computing multilabel evaluation metrics..."):
                evaluator = PipelineEvaluator(pipeline)
                metrics, results, error_details = evaluator.evaluate(
                    examples=benchmark_examples,
                    retrieval_mode=eval_mode,
                    top_k=eval_top_k,
                    use_llm=eval_llm,
                )
                st.session_state["single_eval_metrics"] = metrics
                st.session_state["single_eval_errors"] = error_details

        if "single_eval_metrics" in st.session_state:
            m = st.session_state["single_eval_metrics"]
            errs = st.session_state["single_eval_errors"]

            st.subheader("Evaluation Results")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Micro F1", f"{m.micro_f1:.4f}")
            with k2:
                st.metric("Macro F1", f"{m.macro_f1:.4f}")
            with k3:
                st.metric("Retrieval Recall@1", f"{m.recall_at_1:.4f}")
            with k4:
                st.metric("Retrieval Recall@5", f"{m.recall_at_5:.4f}")

            k5, k6, k7, k8 = st.columns(4)
            with k5:
                st.metric("Mean Reciprocal Rank (MRR)", f"{m.mrr:.4f}")
            with k6:
                st.metric("Coverage", f"{m.coverage * 100:.1f}%")
            with k7:
                st.metric("Abstention Rate", f"{m.abstention_rate * 100:.1f}%")
            with k8:
                st.metric("Latency", f"{m.runtime_seconds:.2f} s")

            with st.expander("Inspect Error Analysis Details (False Positives & Negatives)"):
                st.write(f"**Total False Positives:** {errs['total_false_positives']}")
                st.write(f"**Total False Negatives:** {errs['total_false_negatives']}")
                st.json(errs)

    with eval_tab2:
        st.markdown("Run the controlled ablation study across all 8 configurations.")
        if st.button("🔬 Run All Ablation Benchmarks (A - H)"):
            with st.spinner("Running full ablation suite across all retrieval and fusion methods..."):
                df_ablations = run_ablation_benchmarks(
                    pipeline=pipeline,
                    benchmark_examples=benchmark_examples,
                    output_dir=BASE_DIR / "artifacts",
                )
                st.session_state["ablation_results_df"] = df_ablations

        if "ablation_results_df" in st.session_state:
            st.subheader("Ablation Experiment Comparison Table")
            st.dataframe(st.session_state["ablation_results_df"], use_container_width=True)


# ---------------------------------------------------------
# Page 6: Settings & Offline Status
# ---------------------------------------------------------
elif page == "Settings & Offline Status":
    st.header("Settings & Offline Integrity Status")
    st.markdown("Inspect local environment dependencies, indexes, and offline readiness.")

    diag1, diag2 = st.columns(2)
    with diag1:
        st.subheader("Local Components")
        st.write(f"**Knowledge Base:** `{kb_label}`")
        st.write(f"**Vector Index Path:** `{pipeline.config.paths.vector_index_file}`")
        st.write(f"**BM25 Index Path:** `{pipeline.config.paths.bm25_index_file}`")
        st.write(f"**Embedding Model:** `{pipeline.config.retrieval.embedding_model}`")
        st.write(f"**RRF k-constant:** `{pipeline.config.retrieval.rrf_k}`")

    with diag2:
        st.subheader("Inference Environment")
        st.write(f"**Offline Ready:** `Yes (Zero cloud APIs required)`")
        st.write(f"**Ollama Endpoint:** `{OLLAMA_CHAT_URL}`")
        st.write(f"**RAG-ATT&CK Model:** `{RAG_MODEL}`")
        st.write(f"**Ollama Service Reachable:** `{'Yes' if ollama_ok else 'No'}`")
        if ollama_ok:
            try:
                st.write(
                    f"**Available Local Models:** `{', '.join(pipeline.llm_client.list_available_models())}`"
                )
            except Exception:
                pass

    st.divider()
    st.subheader("Index Maintenance")
    if st.button("🔄 Force Rebuild Local Indexes (BM25 + Vector)"):
        with st.spinner("Rebuilding indexes from knowledge base..."):
            from threatmap.retrieval.index_manager import IndexManager
            mgr = IndexManager(pipeline.config, base_dir=BASE_DIR)
            pipeline.retriever = mgr.build_indices(kb=pipeline.kb, force_rebuild=True)
            st.success("Successfully rebuilt and saved local indexes!")

    st.divider()
    st.subheader("Attribution & Academic Integrity")
    st.markdown(
        """
        * **MITRE ATT&CK®**: Registered trademark of The MITRE Corporation. STIX content licensed under Apache 2.0.
        * **RAG-ATT&CK Reference**: Stijn Schuurman, *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence to MITRE ATT&CK Techniques*, University of Twente (2024).
        * **TRAM Benchmark**: Center for Threat-Informed Defense (CTID).
        """
    )
