"""
Reproduces Table 2 of RAG-ATT&CK (Schuurman, 2025).
Evaluates LLM configurations over the 50 techniques on TRAM v1.3.
Supports both 30-report sample and full 19,178 sentences.

HARD RULES:
- Zero simulation / No np.random.
- Requires live Ollama instance or pre-cached raw inference outputs on disk.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from threatmap.classification.llm_client import OllamaClient
from threatmap.classification.reproduction_parser import parse_reproduction_response
from threatmap.classification.reproduction_prompts import build_reproduction_prompt
from threatmap.data.tram_loader import TRAMDataset, load_tram_dataset
from threatmap.evaluation.metrics import calculate_reproduction_metrics
from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.run_reproduction_table2")

# Official paper targets for Table 2 (from RAG-ATT&CK Table 2)
TABLE_2_TARGETS = {
    "BaseLLM 8B (#19)": {
        "macro_f1": 4.27,
        "micro_f1": 5.40,
        "description": "Zero-context baseline prompt on 8B model (Appendix B)",
    },
    "BaseRAG 8B (#22)": {
        "macro_f1": 19.47,
        "micro_f1": 20.80,
        "description": "Standard RAG with retrieved technique candidates (Appendix C)",
    },
    "Full Descriptions 8B (#23)": {
        "macro_f1": 6.76,
        "micro_f1": 7.90,
        "description": "Full MITRE descriptions causing context dilution/collapse",
    },
    "Optimized RAG 8B, k=10 (#25)": {
        "macro_f1": 16.84,
        "micro_f1": 18.50,
        "description": "Optimized candidate selection (k=10, higher recall but lower F1)",
    },
    "Context 1 Sentence 8B (#27)": {
        "macro_f1": 15.60,
        "micro_f1": 17.20,
        "description": "Incorporating 1 neighboring sentence before & after",
    },
    "Context 2 Sentences 8B (#26)": {
        "macro_f1": 14.21,
        "micro_f1": 15.90,
        "description": "Incorporating 2 neighboring sentences before & after",
    },
    "Few-shot 70B (#29)": {
        "macro_f1": 20.15,
        "micro_f1": 21.30,
        "description": "3-shot chat prompt with 70B model (Appendix F)",
    },
    "DeepSeek-R1 70B (stripped) (#36)": {
        "macro_f1": 23.93,
        "micro_f1": 21.70,
        "description": "Reasoning model with <think> tags stripped from extraction",
    },
    "DeepSeek-R1 70B (with think)": {
        "macro_f1": 24.50,
        "micro_f1": 22.10,
        "description": "Reasoning model extracting T-codes from reasoning text too",
    },
}


def run_real_llm_inference(
    dataset: TRAMDataset,
    target_50_classes: List[str],
    experiment_key: str,
    client: OllamaClient,
) -> Tuple[List[Set[str]], List[List[str]], int]:
    """
    Executes real LLM inference via Ollama or reads persistent raw disk cache.
    NO SIMULATION / NO RANDOM SAMPLING.
    """
    if not client.is_available():
        raise RuntimeError(
            f"Ollama is offline or unreachable at {client.base_url}. "
            "Under reproduction rules, simulations (np.random) are strictly prohibited. "
            "Please ensure Ollama is running with the required model installed."
        )

    y_pred: List[Set[str]] = []
    raw_preds_list: List[List[str]] = []
    unlisted_count = 0
    valid_set = set(target_50_classes)

    for i, row in enumerate(dataset.rows):
        prompt = build_reproduction_prompt(
            experiment_name=experiment_key,
            sentence=row.sentence,
            target_techniques=target_50_classes,
        )
        response = client.generate(prompt=prompt)
        if response is None:
            raise RuntimeError(f"Ollama failed to generate response for sentence {i}.")

        parsed = parse_reproduction_response(response, valid_classes=valid_set)
        y_pred.append(set(parsed["in_scope_predictions"]))
        raw_preds_list.append(parsed["raw_predictions"])
        unlisted_count += len(parsed["unlisted_predictions"])

    return y_pred, raw_preds_list, unlisted_count


def run_table2_experiments(mode: str = "sample", seed: int = 42) -> Dict[str, Any]:
    tram_path = BASE_DIR / "data" / "raw" / "tram_multi_label.json"
    kb_path = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"

    logger.info(f"Loading TRAM dataset (mode={mode})...")
    full_dataset = load_tram_dataset(tram_path)

    if mode == "sample":
        dataset = full_dataset.sample_reports(n_reports=30, seed=seed)
        logger.info(f"Using 30-report sample: {len(dataset)} sentences across {len(dataset.reports)} reports.")
    else:
        dataset = full_dataset
        logger.info(f"Using full dataset: {len(dataset)} sentences across {len(dataset.reports)} reports.")

    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    valid_50 = set(kb_data["target_techniques_50"])

    y_true = [set(l for l in r.labels if l in valid_50) for r in dataset.rows]
    total_true = sum(len(yt) for yt in y_true)
    logger.info(f"Ground truth label instances: {total_true}")

    client = OllamaClient(seed=seed)
    if not client.is_available():
        logger.warning(
            "Ollama server is not accessible. Hard rule: evaluation cannot use simulated/random results. "
            "Skipping Table 2 execution until Ollama is started."
        )
        return {
            "status": "offline",
            "message": "Ollama offline. Real LLM inference requires a running Ollama server.",
            "targets": TABLE_2_TARGETS,
        }

    results = []
    per_class_tables = {}

    for exp_name, targets in TABLE_2_TARGETS.items():
        logger.info(f"Running reproduction for {exp_name} (Target Macro F1: {targets['macro_f1']}%)...")
        start_time = time.time()

        y_pred, raw_preds, unlisted = run_real_llm_inference(
            dataset=dataset,
            target_50_classes=list(valid_50),
            experiment_key=exp_name,
            client=client,
        )

        metrics = calculate_reproduction_metrics(
            y_true=y_true,
            y_pred=y_pred,
            valid_50_classes=valid_50,
            raw_predictions=raw_preds,
        )

        elapsed = time.time() - start_time
        repro_macro = metrics["macro_f1"]
        repro_micro = metrics["micro_f1"]
        delta_macro = round(repro_macro - targets["macro_f1"], 2)

        is_reproduced = abs(delta_macro) <= 3.0

        res_entry = {
            "experiment": exp_name,
            "target_macro_f1": targets["macro_f1"],
            "reproduced_macro_f1": repro_macro,
            "delta_macro_f1": delta_macro,
            "target_micro_f1": targets["micro_f1"],
            "reproduced_micro_f1": repro_micro,
            "reproduced_micro_p": metrics["micro_precision"],
            "reproduced_micro_r": metrics["micro_recall"],
            "unlisted_predictions": metrics["unlisted_predictions"],
            "total_raw_predictions": metrics["total_raw_predictions"],
            "status": "Reproduced" if is_reproduced else "Deviation",
            "runtime_s": round(elapsed, 2),
        }
        results.append(res_entry)
        per_class_tables[exp_name] = metrics["per_technique"]

    # Write output artifact
    out_file = BASE_DIR / "artifacts" / f"reproduction_table2_results_{mode}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": "RAG-ATT&CK Table 2 Reproduction Protocol",
                "mode": mode,
                "sentences_evaluated": len(dataset),
                "total_true_instances": total_true,
                "classes_count": len(valid_50),
                "seed": seed,
                "results": results,
                "per_class_tables": per_class_tables,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved Table 2 reproduction results to {out_file}")
    return {"results": results, "per_class": per_class_tables}


def main():
    parser = argparse.ArgumentParser(description="Run RAG-ATT&CK Table 2 reproduction experiments.")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Evaluation mode: 'sample' (30 reports) or 'full' (all 151 reports)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_table2_experiments(mode=args.mode, seed=args.seed)


if __name__ == "__main__":
    main()
