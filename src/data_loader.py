"""
data_loader.py — Image loading, preprocessing, splitting, and augmentation.

Usage:
    from src.data_loader import load_dataset, get_generators

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()
    train_gen, val_gen = get_generators(X_train, y_train, X_val, y_val)
"""

import os
import sys

# ── Fix import path so 'src' is always found ─────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)
from src.config import (
    DATASET_DIR, IMG_SIZE, CLASS_MAP,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
    BATCH_SIZE, AUGMENTATION
)


# ─────────────────────────────────────────────────────────────────────────────
# Load raw images from disk
# ─────────────────────────────────────────────────────────────────────────────

def load_images_from_disk(folder_path, img_size=IMG_SIZE):
    """
    Load images from a specific directory structure: folder_path/class_name/images.
    """
    images, labels = [], []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        return np.array([]), np.array([])

    for label_name, label_val in CLASS_MAP.items():
        folder = os.path.join(folder_path, label_name)
        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for fname in files:
            try:
                img = load_img(
                    os.path.join(folder, fname),
                    target_size=img_size,
                    color_mode="rgb"
                )
                images.append(img_to_array(img) / 255.0)
                labels.append(label_val)
            except Exception as exc:
                print(f"Error loading {fname}: {exc}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels,  dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Dataset summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(X, y, split_name="Dataset"):
    if len(X) == 0:
        print(f"\n  {split_name}: Empty dataset")
        return
    counts = {label: (y == val).sum() for label, val in CLASS_MAP.items()}
    counts_str = " | ".join([f"{k}: {v}" for k, v in counts.items()])
    print(f"\n  {split_name}: {len(X)} images  |  {counts_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test split
# ─────────────────────────────────────────────────────────────────────────────

def get_train_val_split(X, y, val_size=VAL_SIZE):
    """
    Stratified split into train / val.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=y
    )
    return X_train, X_val, y_train, y_val


# ─────────────────────────────────────────────────────────────────────────────
# Keras data generators (augmentation on train only)
# ─────────────────────────────────────────────────────────────────────────────

def get_generators(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """
    Build augmented train generator and plain validation generator.
    """
    train_datagen = ImageDataGenerator(**AUGMENTATION)
    val_datagen   = ImageDataGenerator()   # no augmentation

    train_gen = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        seed=RANDOM_SEED
    )
    val_gen = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    return train_gen, val_gen


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load everything in one call
# ─────────────────────────────────────────────────────────────────────────────

def load_train_val_data(dataset_dir=DATASET_DIR):
    """
    Load data from Training folder and split into Train/Val.
    """
    train_path = os.path.join(dataset_dir, "Training")
    print(f"Loading Training data from {train_path} ...")
    
    X_full, y_full = load_images_from_disk(train_path)
    
    if len(X_full) == 0:
        raise ValueError(f"No images found in {train_path}.")

    X_train, X_val, y_train, y_val = get_train_val_split(X_full, y_full)
    
    print_summary(X_train, y_train, "Train Set")
    print_summary(X_val,   y_val,   "Val Set  ")
    
    return X_train, X_val, y_train, y_val

def load_test_data(dataset_dir=DATASET_DIR):
    """
    Load data from Testing folder.
    """
    test_path = os.path.join(dataset_dir, "Testing")
    print(f"Loading Testing data from {test_path} ...")
    
    X_test, y_test = load_images_from_disk(test_path)
    
    if len(X_test) == 0:
        raise ValueError(f"No images found in {test_path}.")

    print_summary(X_test, y_test, "Test Set ")
    
    return X_test, y_test
