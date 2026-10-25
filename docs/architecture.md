# ThreatMap System Architecture

## Architectural Overview

ThreatMap is an on-premise, offline-capable NLP system designed to ingest unstructured cybersecurity threat intelligence reports and map extracted adversary behaviors to structured **MITRE ATT&CK** enterprise techniques.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & NLP Preprocessing"]
        A["Threat Report (Text / PDF)"] --> B["Document Parser"]
        B --> C["Cyber-Aware Sentence Splitter"]
        C --> D["Context Window Builder"]
    end

    subgraph Retrieval ["2. Hybrid Retrieval Engine"]
        D --> E1["BM25 Lexical Retriever"]
        D --> E2["Dense Semantic Retriever (Sentence Transformers)"]
        
        KB[("Local MITRE ATT&CK KB (STIX 2.1)")] --> E1
        KB --> E2
        
        E1 --> F["Candidate Rank Lists"]
        E2 --> F
        
        F --> G["Reciprocal Rank Fusion (RRF)"]
        G --> H["Deduplicated Candidate Pool (Top-K)"]
        H --> I["Optional Cross-Encoder Reranker"]
    end

    subgraph Classification ["3. Evidence Grounding & Classification"]
        I --> J{"Local LLM (Ollama) Available?"}
        J -- "Yes" --> K["Prompt Builder (Anti-Injection + Bounded Candidates)"]
        K --> L["Local LLM Inference (e.g. Llama 3 / Mistral)"]
        L --> M["Structured Output Parser"]
        J -- "No / Offline" --> N["Retrieval-Only Candidate Fallback"]
        
        M --> O["Evidence Quote Grounding Validator"]
        N --> O
        KB --> O
    end

    subgraph Assurance ["4. False-Positive Control & Interface"]
        O --> P["Abstention Engine (Confidence & Mitigation Filter)"]
        P --> Q["Verified Structured Prediction"]
        
        Q --> R["Streamlit Professional Dashboard"]
        Q --> S["Audit Exports (CSV / JSON / JSONL)"]
        Q --> T["Evaluation Lab (Multilabel Metrics & Ablations)"]
    end
```

## Architectural Layers

### 1. Ingestion & Preprocessing
* **Document Parser**: Ingests pasted raw text, `.txt` files, and `.pdf` threat bulletins with 10MB input sanitization.
* **Cybersecurity-Aware Sentence Splitter**: Protects technical identifiers with periods (`powershell.exe`, `192.168.1.5`, `T1059.001`, `CVE-2023-38831`, `https://...`) from inadvertent sentence truncation.
* **Context Window Builder**: Associates each target sentence with its predecessor ($S_{i-1}$) and successor ($S_{i+1}$) to retain narrative context.

### 2. Hybrid Retrieval Engine
* **Lexical Retrieval (BM25 Okapi)**: Indexes exact command-line syntax, binary names, and ATT&CK IDs.
* **Semantic Retrieval**: Generates 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`. Computes exact dot-product cosine similarity over unit-normalized embeddings.
* **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$
  Fuses lexical and semantic rankings with parameter $k=60$.
* **Optional Reranker**: Employs a cross-encoder model to score query-candidate pairs directly.

### 3. Classification & Evidence Grounding
* **Strict Candidate Bounding**: The local LLM is restricted to classifying only among retrieved candidates.
* **Anti-Prompt-Injection Architecture**: System prompts instruct the LLM to treat all input report content as untrusted observational evidence, neutralising embedded instructions.
* **Evidence Quote Verification**: Predictions require verbatim textual evidence quotes present in the report.

### 4. False-Positive Control & Abstention
* **Abstention Engine**: Suppresses predictions when similarity falls below calibrated thresholds or when sentences represent non-adversarial text (mitigation advice, disclaimers).
