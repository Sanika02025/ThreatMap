# ThreatMap 2-to-3 Minute Demonstration Script

This script is tailored for a polished, impactful demonstration in an academic presentation, lab viva, or interview.

---

### Step 1: Motivation & Problem Context (30 seconds)
> *"Hello everyone. Today I'm demonstrating **ThreatMap**, an offline-capable, hybrid RAG pipeline for automated MITRE ATT&CK threat intelligence extraction.*
>
> *Security analysts receive hundreds of pages of unstructured incident reports every week. Manually identifying whether an attack used PowerShell, Scheduled Tasks, or LSASS memory dumping is slow and inconsistent. Cloud LLMs cannot be used due to privacy risks with confidential incident data.*
>
> *ThreatMap solves this with an on-premise pipeline combining BM25 lexical search, dense semantic embeddings, Reciprocal Rank Fusion, and evidence-grounded false-positive control."*

---

### Step 2: System Architecture & Ingestion (45 seconds)
> *(Open Streamlit Dashboard -> "Analyze Threat Report" tab)*
> *"Here on the ThreatMap dashboard, I can paste a threat report or upload a TXT or PDF document. Let's load a real-world APT intrusion report: **Operation Cobalt Tempest**.*
>
> *Notice our cybersecurity-aware sentence segmentation: unlike standard NLP splitters that mistakenly break on `powershell.exe`, IP addresses like `192.168.1.5`, or technique IDs like `T1059.001`, ThreatMap protects these technical tokens intact.*
>
> *We configure our retrieval to **Hybrid RRF** with a candidate pool of top 5 techniques and click **Run ThreatMap Pipeline**."*

---

### Step 3: Granular Evidence Inspection (45 seconds)
> *(Navigate to "Results & Evidence" tab)*
> *"In less than a second, ThreatMap processes the report sentence by sentence. Let's look at the results:*
>
> *1. For Sentence 3: 'an embedded macro executed a hidden PowerShell script' -> ThreatMap correctly maps this to **T1059.001 (Command and Scripting Interpreter: PowerShell)**, extracting the exact evidence quote.*
> *2. For Sentence 5: 'dumped credentials from the LSASS process memory' -> ThreatMap accurately maps to **T1003.001 (OS Credential Dumping: LSASS Memory)**.*
> *3. Crucially, look at the final two sentences: recommendations advising administrators to enforce MFA and defensive mitigation actions. ThreatMap's **Abstention Engine** recognizes these as non-adversarial text and abstains, preventing false alarms."*

---

### Step 4: Quantitative Evaluation Lab & Conclusion (30 seconds)
> *(Navigate to "Evaluation Lab" tab)*
> *"Finally, in our Evaluation Lab, we benchmarked ThreatMap against multilabel threat intelligence datasets. As you can see, our hybrid RRF engine achieves a Retrieval Recall@1 of 100% and an MRR of 1.00 on the benchmark suite, processing sentences in under 40 milliseconds on pure CPU.*
>
> *All results can be exported directly to CSV or JSON for integration into SIEM and SOAR platforms. Thank you!"*
