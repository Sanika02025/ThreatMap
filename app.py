"""
ThreatMap: An Offline, Hybrid RAG Pipeline for Automated MITRE ATT&CK Threat Intelligence Extraction.
Interactive Cybersecurity Threat Intelligence Dashboard.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from threatmap.config import load_config
from threatmap.classification.pipeline import ThreatMapPipeline
from threatmap.data.tram_loader import load_benchmark_dataset
from threatmap.evaluation.benchmarks import run_ablation_benchmarks
from threatmap.evaluation.evaluator import PipelineEvaluator
from threatmap.export.csv_export import export_to_csv
from threatmap.export.json_export import export_to_json, export_to_jsonl

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

# Custom CSS for a professional, crisp cybersecurity research interface
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


@st.cache_resource(show_spinner="Initializing ThreatMap Hybrid Pipeline...")
def get_pipeline():
    """Initializes and caches the ThreatMapPipeline instance."""
    config = load_config(base_dir=BASE_DIR)
    return ThreatMapPipeline(config=config, base_dir=BASE_DIR)


pipeline = get_pipeline()

# Load sample threat report if available
SAMPLE_REPORT_PATH = BASE_DIR / "data" / "samples" / "sample_threat_report.txt"
DEFAULT_SAMPLE_TEXT = (
    SAMPLE_REPORT_PATH.read_text(encoding="utf-8")
    if SAMPLE_REPORT_PATH.exists()
    else "The attacker executed a PowerShell command to download secondary malware."
)

# ---------------------------------------------------------
# Sidebar Navigation
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
    st.markdown("### System Health")
    st.write(f"**Loaded Techniques:** `{len(pipeline.kb)}`")
    st.write(f"**Embedding Model:** `all-MiniLM-L6-v2`")
    st.write(f"**BM25 Index:** `Active`")
    ollama_ok = pipeline.llm_client.is_available()
    if ollama_ok:
        st.success("Local LLM (Ollama): Connected")
    else:
        st.info("Local LLM: Offline (Using Retrieval Fallback)")

    st.caption("ThreatMap v0.1.0 • Academic Research Project")


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
                <div class="metric-value" style="font-size:1.3rem;">{'Online' if ollama_ok else 'Retrieval Fallback'}</div>
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
            * **Evidence Grounding**: Strict quotation validation requires predictions to be supported by verbatim text in the report.
            * **Zero Hallucination Guarantee**: Predictions are strictly bounded to verified MITRE ATT&CK IDs in the local knowledge base.
            * **Calibrated Abstention Engine**: Filters out non-adversarial mitigations, disclaimers, and low-confidence inputs.
            * **Local-First & Zero-Cloud**: Operates completely on-premise without transmitting sensitive threat data to third-party APIs.
            """
        )

    st.subheader("Research Grounding")
    st.info(
        """
        **Methodological Foundation**: Inspired by the findings of *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence
        to MITRE ATT&CK Techniques* (Stijn Schuurman, University of Twente, 2024). ThreatMap introduces hybrid RRF ranking,
        adversary-aware context windows, evidence quote verification, and rigorous false-positive abstention.
        """
    )


# ---------------------------------------------------------
# Page 2: Analyze Threat Report
# ---------------------------------------------------------
elif page == "Analyze Threat Report":
    st.header("Analyze Cyber Threat Intelligence Report")
    st.markdown("Input an unstructured threat report or intelligence bulletin to extract MITRE ATT&CK techniques.")

    # Ingestion Controls
    col_input_opt, col_file = st.columns([2, 1])
    with col_input_opt:
        if st.button("📋 Load Operation Cobalt Tempest Sample"):
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

    # Pipeline Execution Settings
    with st.expander("⚙️ Advanced Pipeline Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            retrieval_mode = st.selectbox(
                "Retrieval Strategy",
                ["hybrid_rrf", "hybrid_weighted", "semantic", "lexical"],
                index=0,
                help="Select between Reciprocal Rank Fusion, Weighted Fusion, or standalone retrieval.",
            )
            top_k = st.slider("Candidate Pool (Top-K)", min_value=1, max_value=10, value=5)
        with c2:
            context_window = st.selectbox(
                "Context Window",
                ["target_with_context_and_candidates", "target_with_candidates", "target_only"],
                index=0,
                help="Include neighboring sentences to provide broader context.",
            )
            use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=False)
        with c3:
            use_llm = st.checkbox(
                "Use Local LLM (Ollama)",
                value=False,
                disabled=not ollama_ok,
                help="If disabled or offline, ThreatMap uses the verified candidate retrieval fallback.",
            )
            min_similarity = st.slider("Abstention Threshold", min_value=0.1, max_value=0.6, value=0.30, step=0.05)

    if st.button("🚀 Run ThreatMap Pipeline", type="primary", use_container_width=True):
        if not report_text.strip():
            st.warning("Please provide a threat report before running analysis.")
        else:
            with st.spinner("Processing document sentence-by-sentence with hybrid RAG..."):
                # Temporarily update abstention threshold
                pipeline.abstention_engine.config.min_semantic_similarity = min_similarity
                pipeline.reranker.enabled = use_reranker

                analysis_result = pipeline.analyze_report(
                    content=report_text,
                    filename="Uploaded_Report.txt",
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    use_llm=use_llm,
                    context_strategy=context_window,
                )
                st.session_state["analysis_result"] = analysis_result
                st.success(
                    f"Analysis Complete! Processed {analysis_result.total_sentences} sentences. "
                    f"Found {len(analysis_result.unique_techniques)} unique ATT&CK techniques."
                )

    # Show quick preview of results if available
    if "analysis_result" in st.session_state:
        st.divider()
        st.info("Results are ready! Navigate to the 'Results & Evidence' tab to inspect the detailed mappings.")


# ---------------------------------------------------------
# Page 3: Results & Evidence
# ---------------------------------------------------------
elif page == "Results & Evidence":
    st.header("Threat Intelligence Extraction Results")

    if "analysis_result" not in st.session_state:
        st.warning("No analysis results found. Please analyze a threat report first in the 'Analyze Threat Report' tab.")
    else:
        res = st.session_state["analysis_result"]

        # Top Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Sentences</div>
                    <div class="metric-value">{res.total_sentences}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Sentences with Techniques</div>
                    <div class="metric-value" style="color:#0284c7;">{res.sentences_with_predictions}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Abstained (False-Positive Filter)</div>
                    <div class="metric-value" style="color:#64748b;">{res.abstention_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Unique Techniques</div>
                    <div class="metric-value" style="color:#16a34a;">{len(res.unique_techniques)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Visualizations (Tactic Breakdown)
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
                st.write(f"**Validation Source:** `{'LLM Validated' if meta.get('use_llm') else 'Retrieval Fallback'}`")
                st.write(f"**Document ID:** `{res.document_id}`")
                st.write(f"**Generated:** `{res.timestamp[:19]}`")

        # Sentence Mappings Table
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

        # Filter controls
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            status_filter = st.multiselect("Filter by Status", ["PREDICTED", "ABSTAINED"], default=["PREDICTED", "ABSTAINED"])
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

        # Granular Sentence Inspector
        st.subheader("🔍 Granular Sentence & Evidence Inspector")
        sentence_ids = [s.sentence_id for s in res.sentence_results]
        selected_sid = st.selectbox("Select a sentence to inspect in detail", sentence_ids)
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
                        st.markdown(f"<div class='evidence-box'><strong>Evidence Quote:</strong> \"{p.evidence_quote}\"</div>", unsafe_allow_html=True)
                        st.markdown(f"**Explanation:** {p.explanation}")

            with ins_col2:
                st.markdown("**Top Retrieved Candidate Techniques (Alternative Ranks):**")
                if target_sent.alternative_candidates:
                    cand_data = []
                    for c in target_sent.alternative_candidates:
                        cand_data.append(
                            {
                                "Rank": c.fused_rank,
                                "ID": c.technique_id,
                                "Name": c.technique_name,
                                "Lexical Rank": c.lexical_rank,
                                "Semantic Rank": c.semantic_rank,
                                "Fused Score": c.fused_score,
                            }
                        )
                    st.dataframe(pd.DataFrame(cand_data), use_container_width=True)
                else:
                    st.write("No candidate techniques retrieved.")

        # Export Buttons
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


# ---------------------------------------------------------
# Page 4: MITRE ATT&CK Explorer
# ---------------------------------------------------------
elif page == "MITRE ATT&CK Explorer":
    st.header("MITRE ATT&CK Knowledge Base Explorer")
    st.markdown(f"Explore the verified local knowledge base containing **{len(pipeline.kb)}** active techniques.")

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
            eval_llm = st.checkbox("Validate with Local LLM (if available)", value=False, disabled=not ollama_ok)

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
        st.write(f"**MITRE ATT&CK KB Path:** `{pipeline.config.paths.processed_attack_file}`")
        st.write(f"**Vector Index Path:** `{pipeline.config.paths.vector_index_file}`")
        st.write(f"**BM25 Index Path:** `{pipeline.config.paths.bm25_index_file}`")
        st.write(f"**Embedding Model:** `{pipeline.config.retrieval.embedding_model}`")
        st.write(f"**RRF k-constant:** `{pipeline.config.retrieval.rrf_k}`")

    with diag2:
        st.subheader("Inference Environment")
        st.write(f"**Offline Ready:** `Yes (Zero cloud APIs required)`")
        st.write(f"**Ollama Endpoint:** `{pipeline.config.classification.ollama_base_url}`")
        st.write(f"**Ollama Service Reachable:** `{'Yes' if ollama_ok else 'No (Retrieval Fallback Active)'}`")
        if ollama_ok:
            st.write(f"**Available Local Models:** `{', '.join(pipeline.llm_client.list_available_models())}`")

    st.divider()
    st.subheader("Index Maintenance")
    if st.button("🔄 Force Rebuild Local Indexes (BM25 + FAISS/Numpy)"):
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
