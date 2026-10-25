# Evaluation Datasets and Benchmark Protocol

## Overview

This directory houses the evaluation datasets for ThreatMap. The evaluation methodology follows the framework outlined in:
* **RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence to MITRE ATT&CK Techniques** (Stijn Schuurman, University of Twente, 2024).
* **TRAM (Threat Report ATT&CK Mapping)** v1.3 by the Center for Threat-Informed Defense (CTID).

## TRAM v1.3 Dataset Context

The research paper utilizes the TRAM v1.3 multilabel dataset consisting of 5,089 annotated CTI sentences covering 50 common ATT&CK techniques/sub-techniques.
In accordance with research integrity standards:
1. ThreatMap does not fabricate results or claim performance on datasets not loaded in the local environment.
2. A hand-verified, high-quality demonstration benchmark (`tram_sample_benchmark.json`) is included for offline verification, testing, and academic project demos.
3. If the complete 5,089-sentence TRAM dataset is loaded by the user, the benchmark runner seamlessly processes the full dataset using the exact same schema.

## Schema Format

The benchmark dataset format uses standard JSON/JSONL with multilabel support:

```json
[
  {
    "id": "TRAM-DEMO-001",
    "text": "The adversary utilized PowerShell commands to download and execute secondary malware payloads.",
    "labels": ["T1059.001", "T1105"],
    "split": "test",
    "source_report": "Operation Cobalt Tempest"
  }
]
```

## Evaluated Metrics

* **Retrieval Recall@k** (k=1, 3, 5, 10): Measures the fraction of true technique IDs present within the top-k retrieved candidates.
* **Mean Reciprocal Rank (MRR)**: Evaluates the rank position of the first relevant technique.
* **Micro Precision, Recall, F1**: Global metrics aggregated across all sentence-technique pairs.
* **Macro Precision, Recall, F1**: Unweighted average across all technique classes, highlighting performance on rare classes.
* **Coverage**: Proportion of sentences where the system produces a non-abstaining prediction.
* **Abstention Rate**: Proportion of sentences where the system appropriately abstains due to low confidence or absence of adversary behavior.
