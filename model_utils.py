#!/usr/bin/env python3
"""
Helpers for training and evaluating the simple food classifier.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = 32
FEATURES = IMAGE_SIZE * IMAGE_SIZE * 3
CLASSES = ["pizza", "steak", "sushi"]
MODEL_METRICS_PATH = Path("results/model_metrics.json")
WEIGHTS_PATH = Path("matrices/W.bin")


def load_matrix(path: str | Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(-1, FEATURES)


def load_labels(path: str | Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32)


def load_weights(path: str | Path = WEIGHTS_PATH) -> np.ndarray:
    weights = np.fromfile(path, dtype=np.float32)
    return weights.reshape(FEATURES, len(CLASSES))


def preprocess_image(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.array(image, dtype=np.float32) / 255.0
    return array.flatten()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def train_linear_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    reg_strength: float = 10.0,
) -> np.ndarray:
    """
    Train a deterministic linear classifier with dual-form ridge regression.
    """
    x64 = x_train.astype(np.float64, copy=False)
    targets = np.eye(len(CLASSES), dtype=np.float64)[y_train]
    gram = x64 @ x64.T + reg_strength * np.eye(x64.shape[0], dtype=np.float64)
    alpha = np.linalg.solve(gram, targets)
    weights = x64.T @ alpha
    return weights.astype(np.float32)


def evaluate_classifier(
    x_data: np.ndarray,
    y_data: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    logits = x_data @ weights
    predictions = logits.argmax(axis=1).astype(np.int32)
    accuracy = float((predictions == y_data).mean())

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int32)
    for actual, predicted in zip(y_data, predictions):
        confusion[int(actual), int(predicted)] += 1

    per_class = {}
    for index, class_name in enumerate(CLASSES):
        total = int((y_data == index).sum())
        correct = int(confusion[index, index])
        per_class[class_name] = {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total) if total else 0.0,
        }

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "predictions": predictions.tolist(),
    }


def predict_from_vector(vector: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    start = time.perf_counter()
    logits = vector.astype(np.float32) @ weights.astype(np.float32)
    probabilities = softmax(logits.astype(np.float64))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_index": predicted_index,
        "predicted_class": CLASSES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "elapsed_ms": elapsed_ms,
    }


def predict_image(image_path: str | Path, weights_path: str | Path = WEIGHTS_PATH) -> dict[str, object]:
    vector = preprocess_image(image_path)
    weights = load_weights(weights_path)
    result = predict_from_vector(vector, weights)
    result["image_path"] = str(image_path)
    image_path = Path(image_path)
    parent_name = image_path.parent.name.lower()
    if parent_name in CLASSES:
        actual_index = CLASSES.index(parent_name)
        result["actual_class"] = parent_name
        result["is_correct"] = bool(result["predicted_index"] == actual_index)
    return result


def workload_macs(sample_count: int) -> int:
    return int(sample_count) * FEATURES * len(CLASSES)


def save_model_metrics(metrics: dict[str, object], path: Path = MODEL_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def load_model_metrics(path: Path = MODEL_METRICS_PATH) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())
