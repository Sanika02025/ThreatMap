"""
generate_figures.py — Generate Fig. 2 and Fig. 4 from TRAM data.

Fig. 2: Distribution of labels per sentence (for labelled sentences in 50-class scope).
        X = number of gold labels, Y = count of sentences.

Fig. 4: Class distribution of 5,143 label instances across 50 techniques.
        X = technique ID (sorted by frequency desc), Y = count.

Saves to:
  artifacts/fig2_label_distribution.png
  artifacts/fig4_class_distribution.png

Hard rules: reads only from data files, no randomness, matplotlib only.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
TRAM_PATH = BASE_DIR / "data" / "tram2-data" / "multi_label.json"
KB_PATH = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"
OUT_DIR = BASE_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG2_PATH = OUT_DIR / "fig2_label_distribution.png"
FIG4_PATH = OUT_DIR / "fig4_class_distribution.png"


def load_data():
    with open(TRAM_PATH, "r", encoding="utf-8") as f:
        tram = json.load(f)
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    valid_50 = {t["technique_id"] for t in kb["techniques"]}
    tech_names = {t["technique_id"]: t["technique_name"] for t in kb["techniques"]}
    return tram, valid_50, tech_names


def make_fig2(tram, valid_50):
    """Fig. 2 — Distribution of gold labels per sentence (50-class scope)."""
    label_counts = []
    for row in tram:
        labels = [l for l in (row.get("labels") or []) if l in valid_50]
        if labels:  # only labelled sentences
            label_counts.append(len(labels))

    counter = Counter(label_counts)
    max_labels = max(counter.keys())
    xs = list(range(1, max_labels + 1))
    ys = [counter.get(x, 0) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(xs, ys, color="#2563EB", edgecolor="white", linewidth=0.5)

    # Annotate bars
    for bar, count in zip(bars, ys):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                str(count),
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xlabel("Number of gold technique labels per sentence", fontsize=12)
    ax.set_ylabel("Number of sentences", fontsize=12)
    ax.set_title(
        "Fig. 2: Label Count Distribution (4,070 in-scope sentences, 50 ATT&CK techniques)",
        fontsize=11, pad=12
    )
    ax.set_xticks(xs)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    total_sent = sum(ys)
    total_labels = sum(x * y for x, y in zip(xs, ys))
    ax.annotate(
        f"Total labelled sentences: {total_sent:,}\nTotal label instances: {total_labels:,}",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8)
    )

    fig.tight_layout()
    fig.savefig(FIG2_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig. 2 saved to {FIG2_PATH}")
    print(f"  Labelled sentences: {total_sent}, label instances: {total_labels}")
    return total_sent, total_labels


def make_fig4(tram, valid_50, tech_names):
    """Fig. 4 — Class distribution of 5,143 labels across 50 techniques."""
    class_counts: Counter = Counter()
    for row in tram:
        for lbl in (row.get("labels") or []):
            if lbl in valid_50:
                class_counts[lbl] += 1

    total = sum(class_counts.values())
    # Sort by frequency descending
    sorted_items = sorted(class_counts.items(), key=lambda x: -x[1])
    tids = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(16, 6))
    x_pos = np.arange(len(tids))
    bars = ax.bar(x_pos, counts, color="#059669", edgecolor="white", linewidth=0.3, width=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tids, rotation=90, fontsize=7)
    ax.set_xlabel("Technique ID (sorted by frequency, desc)", fontsize=11)
    ax.set_ylabel("Number of label instances", fontsize=11)
    ax.set_title(
        f"Fig. 4: Class Distribution of {total:,} Label Instances across 50 ATT&CK Techniques (TRAM v1.3)",
        fontsize=11, pad=12
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate top-5
    for i in range(min(5, len(tids))):
        ax.text(
            x_pos[i], counts[i] + 1,
            tids[i], ha="center", va="bottom", fontsize=6, rotation=90
        )

    fig.tight_layout()
    fig.savefig(FIG4_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig. 4 saved to {FIG4_PATH}")
    print(f"  Total label instances: {total} (expected 5,143)")
    print(f"  Classes with >=1 instance: {len(class_counts)}/50")
    print(f"  Most frequent: {tids[0]} ({counts[0]}), Least: {tids[-1]} ({counts[-1]})")
    return total


def main():
    print("=" * 60)
    print("Generating Figures 2 & 4 from TRAM data")
    print("=" * 60)
    tram, valid_50, tech_names = load_data()
    print(f"Loaded {len(tram)} rows. Valid classes: {len(valid_50)}")

    make_fig2(tram, valid_50)
    make_fig4(tram, valid_50, tech_names)
    print("\nDone.")


if __name__ == "__main__":
    main()

