import os
from pathlib import Path

import numpy as np
from PIL import Image

from model_utils import CLASSES, evaluate_classifier, save_model_metrics, train_linear_classifier

IMAGE_SIZE = 32       # Resize all images to 32x32 for faster GEM5 runs
FLAT_DIM   = IMAGE_SIZE * IMAGE_SIZE * 3   # = 12288 per image
CLASS_MAP  = {c: i for i, c in enumerate(CLASSES)}

data_path  = Path("data/pizza_steak_sushi")


def balanced_subset_indices(labels: np.ndarray, subset_size: int) -> np.ndarray:
    class_indices = [np.where(labels == class_id)[0] for class_id in range(len(CLASSES))]
    selected = []
    while len(selected) < subset_size:
        made_progress = False
        for class_id, indices in enumerate(class_indices):
            if len(selected) >= subset_size:
                break
            take_index = sum(labels[idx] == class_id for idx in selected)
            if take_index < len(indices):
                selected.append(int(indices[take_index]))
                made_progress = True
        if not made_progress:
            break
    return np.array(selected[:subset_size], dtype=np.int32)

def load_split(split: str):
    """Load all images from a split (train/test) into a float32 matrix."""
    images, labels = [], []
    split_path = data_path / split
    for cls in CLASSES:
        cls_path = split_path / cls
        if not cls_path.exists():
            print(f"  WARNING: {cls_path} not found, skipping")
            continue
        for img_file in sorted(cls_path.glob("*.jpg")):
            img = Image.open(img_file).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 255.0   # Normalize [0,1]
            images.append(arr.flatten())                      # 12288-dim vector
            labels.append(CLASS_MAP[cls])
    X = np.array(images, dtype=np.float32)   # (N, 12288)
    y = np.array(labels, dtype=np.int32)     # (N,)
    return X, y

print("Extracting training matrices...")
X_train, y_train = load_split("train")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

print("Extracting test matrices...")
X_test,  y_test  = load_split("test")
print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

# Save as binary files for C++ / GEM5 consumption
os.makedirs("matrices", exist_ok=True)
X_train.tofile("matrices/X_train.bin")
y_train.tofile("matrices/y_train.bin")
X_test.tofile("matrices/X_test.bin")
y_test.tofile("matrices/y_test.bin")

# Train a deterministic linear classifier so we can report train/test accuracy
print("\nTraining linear food classifier...")
W = train_linear_classifier(X_train, y_train, reg_strength=10.0)
train_metrics = evaluate_classifier(X_train, y_train, W)
test_metrics = evaluate_classifier(X_test, y_test, W)

# Also save a balanced test subset for the GEM5 matmul workload
SUBSET = 24
subset_idx = balanced_subset_indices(y_test, SUBSET)
X_sub = X_test[subset_idx]
y_sub = y_test[subset_idx]
subset_metrics = evaluate_classifier(X_sub, y_sub, W)

np.save("matrices/X_sub.npy", X_sub)
np.save("matrices/W.npy",     W)
X_sub.tofile("matrices/X_sub.bin")
W.tofile("matrices/W.bin")
y_sub.tofile("matrices/y_sub.bin")

model_metrics = {
    "classes": CLASSES,
    "train_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "gem5_subset_samples": int(len(y_sub)),
    "train_accuracy": train_metrics["accuracy"],
    "test_accuracy": test_metrics["accuracy"],
    "gem5_subset_accuracy": subset_metrics["accuracy"],
    "train": train_metrics,
    "test": test_metrics,
    "gem5_subset": subset_metrics,
}
save_model_metrics(model_metrics)

print(f"\nSaved matrices to matrices/")
print(f"  X_sub  ({X_sub.shape[0]} x {X_sub.shape[1]}): {X_sub.shape}")
print(f"  W      ({W.shape[0]} x {W.shape[1]}):  {W.shape}")
print(f"  Matrix multiply C = X_sub @ W -> shape {(X_sub @ W).shape}")
print(f"  GEM5 subset labels:  {[int((y_sub == idx).sum()) for idx in range(len(CLASSES))]}")
print(f"  Train accuracy:     {train_metrics['accuracy'] * 100:.2f}%")
print(f"  Test accuracy:      {test_metrics['accuracy'] * 100:.2f}%")
print(f"  GEM5 subset acc.:   {subset_metrics['accuracy'] * 100:.2f}%")
print("\nSaved: results/model_metrics.json")
print("\nMatrices ready for C++ / GEM5!")
