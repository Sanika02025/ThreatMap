"""
Downloads and normalizes official MITRE ATT&CK Enterprise STIX data.
Usage:
    python scripts/download_attack_data.py
"""

import json
import sys
from pathlib import Path
import requests

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from threatmap.config import load_config
from threatmap.data.attack_loader import parse_attack_stix_bundle
from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.download_attack_data")


def main():
    config = load_config(base_dir=BASE_DIR)
    raw_path = BASE_DIR / config.paths.raw_attack_file
    processed_path = BASE_DIR / config.paths.processed_attack_file
    url = config.paths.attack_stix_url

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading official MITRE ATT&CK STIX bundle from: {url}")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        stix_data = response.json()

        logger.info(f"Saving raw STIX data to {raw_path}...")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(stix_data, f)

        logger.info("Parsing STIX bundle into normalized knowledge base...")
        kb = parse_attack_stix_bundle(stix_data)
        kb.save_to_json(processed_path)

        logger.info(
            f"Successfully processed {len(kb)} ATT&CK techniques. "
            f"Normalized data saved to {processed_path}"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading ATT&CK STIX data: {e}")
        logger.info(
            "System will continue operating offline using the verified miniature ATT&CK fixture."
        )
    except Exception as e:
        logger.error(f"Error parsing ATT&CK data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
