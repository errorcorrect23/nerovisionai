"""
model.py — CNN architecture for Brain MRI Tumor Detection.

The network is built entirely from scratch (no pretrained weights).

Architecture summary:
  Block 1:  Conv2D(32)  x2 → BatchNorm → ReLU → MaxPool(2,2) → Dropout(0.25)
  Block 2:  Conv2D(64)  x2 → BatchNorm → ReLU → MaxPool(2,2) → Dropout(0.25)
  Block 3:  Conv2D(128) x2 → BatchNorm → ReLU → MaxPool(2,2) → Dropout(0.30)
  Head:     Flatten → Dense(256, L2) → BatchNorm → ReLU → Dropout(0.50)
            → Dense(1, sigmoid)

Usage:
    from src.model import build_cnn
    model = build_cnn()
    model.summary()
"""

import os
import sys

# ── Fix import path so 'src' is always found ─────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from src.config import INPUT_SHAPE, LEARNING_RATE, L2_REG, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Reusable conv block helper
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(x, filters, dropout_rate):
    """
    Two Conv2D layers with BatchNorm + ReLU, followed by MaxPool and Dropout.

    Args:
        x:            Input tensor.
        filters:      Number of convolutional filters.
        dropout_rate: Fraction of units to drop after pooling.

    Returns:
        Output tensor after the block.
    """
    for _ in range(2):
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(input_shape=INPUT_SHAPE):
    """
    Build and return the CNN model using MobileNetV2 (Transfer Learning).
    
    Args:
        input_shape: Tuple (H, W, C) — defaults to config.INPUT_SHAPE.
    """
    inputs = layers.Input(shape=input_shape, name="mri_input")
    
    # The data loader scales images to [0, 1]. MobileNetV2 expects [-1, 1].
    x = layers.Lambda(lambda t: t * 2.0 - 1.0, name="rescaling")(inputs)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    # Freeze the pre-trained weights so we only train our classification head.
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(
        128, 
        activation="relu", 
        kernel_regularizer=regularizers.l2(L2_REG),
        name="fc_128"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="BrainTumorMobileNetV2")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Compile helper
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile the model with Adam, binary cross-entropy, and common metrics.

    Args:
        model:         A tf.keras.Model instance.
        learning_rate: Initial learning rate for Adam.

    Returns:
        The same model, now compiled.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    m = build_cnn()
    compile_model(m)
    m.summary()
