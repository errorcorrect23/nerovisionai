"""
config.py — Central configuration for the Brain MRI Tumor Detection project.
All hyperparameters and paths are defined here so every module stays in sync.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "best_model.h5")
PLOT_PATH   = os.path.join(OUTPUT_DIR, "training_plots.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Image settings ─────────────────────────────────────────────────────────
IMG_SIZE    = (128, 128)   # (height, width)
IMG_CHANNELS = 3           # RGB
INPUT_SHAPE = (*IMG_SIZE, IMG_CHANNELS)

# ── Class mapping ──────────────────────────────────────────────────────────
CLASS_NAMES   = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_MAP     = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES   = len(CLASS_NAMES)

# ── Data split ─────────────────────────────────────────────────────────────
TEST_SIZE  = 0.15   # 15% test
VAL_SIZE   = 0.15   # 15% validation (from remaining after test split)
RANDOM_SEED = 42

# ── Training hyperparameters ───────────────────────────────────────────────
BATCH_SIZE     = 32
EPOCHS         = 30
LEARNING_RATE  = 1e-3
L2_REG         = 1e-4

# ── Augmentation settings ──────────────────────────────────────────────────
AUGMENTATION = dict(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest",
)

# ── Callbacks ──────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 8
REDUCE_LR_PATIENCE   = 4
REDUCE_LR_FACTOR     = 0.5
REDUCE_LR_MIN        = 1e-6

# ── Decision threshold ─────────────────────────────────────────────────────
THRESHOLD = 0.50
