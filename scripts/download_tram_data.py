"""
Downloads and verifies the official TRAM v1.3 multi-label dataset at pinned commit f29793d8.
URL: https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/f29793d8d665f7f552898696e00065ef24a29a20/data/tram2-data/multi_label.json
"""

import json
from pathlib import Path
import urllib.request
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.download_tram_data")

PINNED_COMMIT = "f29793d8d665f7f552898696e00065ef24a29a20"
TRAM_URL = f"https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/{PINNED_COMMIT}/data/tram2-data/multi_label.json"
EXPECTED_SENTENCES = 19178
EXPECTED_REPORTS = 151
EXPECTED_LABELED = 4070
EXPECTED_UNLABELED = 15108
EXPECTED_LABEL_INSTANCES = 5143
EXPECTED_CLASSES = 50


def download_and_verify_tram(dest_path: Path) -> dict:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        logger.info(f"Downloading TRAM dataset from {TRAM_URL} to {dest_path}...")
        urllib.request.urlretrieve(TRAM_URL, dest_path)
        logger.info(f"Downloaded {dest_path.stat().st_size} bytes.")
    else:
        logger.info(f"TRAM multi-label dataset already exists at {dest_path}")

    with open(dest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_sentences = len(data)
    doc_titles = {d.get("doc_title") for d in data}
    labeled = [d for d in data if d.get("labels")]
    unlabeled = [d for d in data if not d.get("labels")]
    all_labels = [lbl for d in data for lbl in d.get("labels", [])]
    unique_classes = set(all_labels)

    logger.info("Verifying TRAM dataset invariants:")
    logger.info(f"  Total sentences: {total_sentences} (expected: {EXPECTED_SENTENCES})")
    logger.info(f"  Unique reports: {len(doc_titles)} (expected: {EXPECTED_REPORTS})")
    logger.info(f"  Labeled rows: {len(labeled)} (expected: {EXPECTED_LABELED})")
    logger.info(f"  Unlabeled rows: {len(unlabeled)} (expected: {EXPECTED_UNLABELED})")
    logger.info(f"  Total label instances: {len(all_labels)} (expected: {EXPECTED_LABEL_INSTANCES})")
    logger.info(f"  Unique technique classes: {len(unique_classes)} (expected: {EXPECTED_CLASSES})")

    assert total_sentences == EXPECTED_SENTENCES, f"Sentence count mismatch: {total_sentences} vs {EXPECTED_SENTENCES}"
    assert len(doc_titles) == EXPECTED_REPORTS, f"Report count mismatch: {len(doc_titles)} vs {EXPECTED_REPORTS}"
    assert len(all_labels) == EXPECTED_LABEL_INSTANCES, f"Label instance mismatch: {len(all_labels)} vs {EXPECTED_LABEL_INSTANCES}"
    assert len(unique_classes) == EXPECTED_CLASSES, f"Unique classes mismatch: {len(unique_classes)} vs {EXPECTED_CLASSES}"

    logger.info("TRAM dataset verification passed successfully!")
    return {
        "total_sentences": total_sentences,
        "reports": len(doc_titles),
        "labeled_sentences": len(labeled),
        "unlabeled_sentences": len(unlabeled),
        "label_instances": len(all_labels),
        "unique_classes": sorted(list(unique_classes)),
    }


def main():
    dest = BASE_DIR / "data" / "raw" / "tram_multi_label.json"
    download_and_verify_tram(dest)


if __name__ == "__main__":
    main()
