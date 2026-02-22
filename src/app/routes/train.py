import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.modules.train_module_res50 import train_model

router = APIRouter()


class TrainRequest(BaseModel):
    batch_size: int = Field(..., gt=0, alias="batchSize", description="Batch size for training")
    epochs: int = Field(..., gt=0, description="Number of epochs for training")
    learning_rate: float = Field(..., gt=0, alias="learningRate", description="Learning rate for the optimizer")


def _load_training_metrics() -> Dict[str, List[float]]:
    src_root = Path(__file__).resolve().parents[2]
    logs_dir = src_root / "logs"
    candidate_paths = [
        logs_dir / "history.json",
        logs_dir / "training_history.json",
    ]

    history_path = next((p for p in candidate_paths if p.exists()), None)
    if history_path is None:
        raise HTTPException(status_code=404, detail="Training metrics not found")

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {e}")

    if not isinstance(metrics, dict):
        raise HTTPException(status_code=500, detail="Training metrics format is invalid")
    return metrics


def _load_json_file(path: Path, not_found_detail: str):
    if not path.exists():
        raise HTTPException(status_code=404, detail=not_found_detail)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading {path.name}: {e}")


def _normalize_rows(matrix: List[List[int]]) -> List[List[float]]:
    normalized = []
    for row in matrix:
        row_sum = float(sum(row))
        if row_sum <= 0:
            normalized.append([0.0 for _ in row])
        else:
            normalized.append([float(v) / row_sum for v in row])
    return normalized


def _build_test_report_plot_ready(report: Dict[str, Any], misclassified: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_class = report.get("per_class", [])
    confusion_matrix = report.get("confusion_matrix", [])

    recall_sorted = sorted(per_class, key=lambda r: float(r.get("recall", 0.0)))
    f1_sorted = sorted(per_class, key=lambda r: float(r.get("f1", 0.0)))

    mis_pair_counter = Counter()
    for item in misclassified:
        true_label = str(item.get("true_label", "Unknown"))
        pred_label = str(item.get("pred_label", "Unknown"))
        mis_pair_counter[f"{true_label} -> {pred_label}"] += 1
    top_pairs = mis_pair_counter.most_common(20)

    return {
        "confusion_matrix": {
            "chartType": "heatmap",
            "labels": [str(r.get("label", "")) for r in per_class],
            "values": confusion_matrix,
            "rowNormalizedValues": _normalize_rows(confusion_matrix),
        },
        "per_class_recall": {
            "chartType": "bar",
            "labels": [str(r.get("label", "")) for r in recall_sorted],
            "values": [float(r.get("recall", 0.0)) for r in recall_sorted],
            "support": [int(r.get("support", 0)) for r in recall_sorted],
        },
        "per_class_precision_f1": {
            "chartType": "grouped_bar",
            "labels": [str(r.get("label", "")) for r in f1_sorted],
            "series": [
                {
                    "name": "precision",
                    "values": [float(r.get("precision", 0.0)) for r in f1_sorted],
                },
                {
                    "name": "f1",
                    "values": [float(r.get("f1", 0.0)) for r in f1_sorted],
                },
            ],
        },
        "misclassified_pairs": {
            "chartType": "barh",
            "labels": [pair for pair, _ in top_pairs],
            "values": [count for _, count in top_pairs],
        },
    }


def _build_plot_ready(metrics: Dict[str, List[float]]) -> Dict[str, object]:
    max_len = max((len(v) for v in metrics.values() if isinstance(v, list)), default=0)
    epochs = list(range(1, max_len + 1))
    series = []

    def infer_chart_type(metric_name: str, values: List[float]) -> str:
        # Metrics logged once at test-time are easier to read as bars.
        if metric_name.startswith("test_") or len(values) <= 1:
            return "bar"
        return "line"

    for name, values in metrics.items():
        if isinstance(values, list):
            series.append(
                {
                    "name": name,
                    "values": values,
                    "chartType": infer_chart_type(name, values),
                }
            )
    return {"epochs": epochs, "series": series}


@router.post("/start")
async def start_training(
    params: TrainRequest,
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(train_model, params.epochs, params.batch_size, params.learning_rate)
    return {"message": "Training started"}


@router.post("/stop")
async def stop_training():
    return {"message": "Training stop request sent"}


@router.get("/metrics")
async def get_training_metrics():
    metrics = _load_training_metrics()
    return {
        "history": metrics,
        "plot_ready": _build_plot_ready(metrics),
    }


@router.get("/metrics/test-report")
async def get_test_report_metrics():
    src_root = Path(__file__).resolve().parents[2]
    logs_dir = src_root / "logs"
    report_path = logs_dir / "classification_report.json"
    misclassified_path = logs_dir / "top_misclassified.json"

    report = _load_json_file(report_path, "classification_report.json not found")
    misclassified = _load_json_file(misclassified_path, "top_misclassified.json not found")

    if not isinstance(report, dict):
        raise HTTPException(status_code=500, detail="classification_report.json format is invalid")
    if not isinstance(misclassified, list):
        raise HTTPException(status_code=500, detail="top_misclassified.json format is invalid")

    summary = {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro_avg", {}).get("f1", 0.0)),
        "weighted_f1": float(report.get("weighted_avg", {}).get("f1", 0.0)),
        "num_test_samples_scored": int(report.get("num_test_samples_scored", 0)),
        "num_misclassified": int(report.get("num_misclassified", len(misclassified))),
    }

    return {
        "summary": summary,
        "plot_ready": _build_test_report_plot_ready(report, misclassified),
    }


@router.get("/metrics/plot")
async def get_training_metrics_plot():
    metrics = _load_training_metrics()

    epochs = list(range(1, max((len(v) for v in metrics.values() if isinstance(v, list)), default=0) + 1))
    if not epochs:
        raise HTTPException(status_code=404, detail="No metric points available to plot")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

    loss = metrics.get("loss", [])
    val_loss = metrics.get("val_loss", [])
    if loss:
        axes[0].plot(range(1, len(loss) + 1), loss, label="loss")
    if val_loss:
        axes[0].plot(range(1, len(val_loss) + 1), val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    acc = metrics.get("accuracy", [])
    val_acc = metrics.get("val_accuracy", [])
    if acc:
        axes[1].plot(range(1, len(acc) + 1), acc, label="accuracy")
    if val_acc:
        axes[1].plot(range(1, len(val_acc) + 1), val_acc, label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
