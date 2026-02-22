import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _wrap_labels(labels, width=24):
    wrapped = []
    for label in labels:
        wrapped.append("\n".join(textwrap.wrap(str(label), width=width)) or str(label))
    return wrapped


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_confusion_matrix(report: dict, out_path: Path):
    labels = [row["label"] for row in report["per_class"]]
    cm = np.array(report["confusion_matrix"], dtype=np.int64)
    support = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, support, out=np.zeros_like(cm, dtype=float), where=support > 0)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Test Confusion Matrix (Row-normalized)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    idx = np.arange(len(labels))
    wrapped = _wrap_labels(labels)
    ax.set_xticks(idx)
    ax.set_xticklabels(wrapped, rotation=90, fontsize=7)
    ax.set_yticks(idx)
    ax.set_yticklabels(wrapped, fontsize=7)

    # Write normalized value only for cells with at least one sample on true class row.
    for i in range(cm.shape[0]):
        if support[i, 0] == 0:
            continue
        for j in range(cm.shape[1]):
            v = cm_norm[i, j]
            if v <= 0:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of true-class samples")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_recall(report: dict, out_path: Path):
    rows = sorted(report["per_class"], key=lambda x: x["recall"])
    labels = [r["label"] for r in rows]
    values = [r["recall"] for r in rows]
    support = [r["support"] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=120)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color="#2a9d8f")
    ax.set_title("Per-class Recall (sorted low to high)")
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap_labels(labels, width=28), fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for i, bar in enumerate(bars):
        ax.text(
            min(bar.get_width() + 0.01, 0.98),
            bar.get_y() + bar.get_height() / 2.0,
            f"{values[i]:.2f} (n={support[i]})",
            va="center",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_precision_f1(report: dict, out_path: Path):
    rows = sorted(report["per_class"], key=lambda x: x["f1"])
    labels = [r["label"] for r in rows]
    precision = [r["precision"] for r in rows]
    f1 = [r["f1"] for r in rows]

    x = np.arange(len(labels))
    width = 0.42

    fig, ax = plt.subplots(figsize=(15, 7), dpi=120)
    ax.bar(x - width / 2, precision, width=width, label="Precision", color="#457b9d")
    ax.bar(x + width / 2, f1, width=width, label="F1", color="#e76f51")
    ax.set_title("Per-class Precision and F1")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(_wrap_labels(labels, width=18), rotation=90, fontsize=7)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_misclassified_pairs(misclassified: list[dict], out_path: Path, top_k=20):
    pair_counter = Counter()
    for row in misclassified:
        pair = f"{row.get('true_label', 'Unknown')} -> {row.get('pred_label', 'Unknown')}"
        pair_counter[pair] += 1

    top_pairs = pair_counter.most_common(top_k)
    if not top_pairs:
        top_pairs = [("No misclassified samples", 0)]

    labels = [p[0] for p in top_pairs]
    values = [p[1] for p in top_pairs]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=120)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color="#f4a261")
    ax.set_title(f"Top {min(top_k, len(labels))} Misclassified Label Pairs")
    ax.set_xlabel("Count")
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap_labels(labels, width=44), fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2.0, str(values[i]), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot test evaluation charts from classification_report.json")
    parser.add_argument(
        "--report",
        default="src/logs/classification_report.json",
        help="Path to classification_report.json",
    )
    parser.add_argument(
        "--misclassified",
        default="src/logs/top_misclassified.json",
        help="Path to top_misclassified.json",
    )
    parser.add_argument(
        "--out-dir",
        default="src/logs/plots",
        help="Output folder for generated plots",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    misclassified_path = Path(args.misclassified)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _read_json(report_path)
    misclassified = _read_json(misclassified_path) if misclassified_path.exists() else []

    out_confusion = out_dir / "test_confusion_matrix.png"
    out_recall = out_dir / "test_per_class_recall.png"
    out_pr_f1 = out_dir / "test_per_class_precision_f1.png"
    out_mis = out_dir / "test_top_misclassified_pairs.png"

    _plot_confusion_matrix(report, out_confusion)
    _plot_per_class_recall(report, out_recall)
    _plot_precision_f1(report, out_pr_f1)
    _plot_misclassified_pairs(misclassified, out_mis)

    print("Generated plots:")
    print(f"- {out_confusion}")
    print(f"- {out_recall}")
    print(f"- {out_pr_f1}")
    print(f"- {out_mis}")


if __name__ == "__main__":
    main()
