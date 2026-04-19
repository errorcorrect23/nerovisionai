"""
predict.py — Inference on new MRI images using the saved model.

Usage (CLI):
    python src/predict.py --image path/to/scan.jpg
    python src/predict.py --image path/to/scan.jpg --threshold 0.6

Usage (Python API):
    from src.predict import load_model, predict_image, predict_batch

    model = load_model()
    result = predict_image("scan.jpg", model)
    print(result)
"""

import os
import sys

# ── Suppress TensorFlow & absl warnings ─────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── Fix import path so 'src' is always found ─────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.config import MODEL_PATH, IMG_SIZE, CLASS_NAMES, THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path=MODEL_PATH):
    """
    Load the saved Keras model from disk.

    Args:
        model_path: Path to the .h5 model file.

    Returns:
        Compiled tf.keras.Model.

    Raises:
        FileNotFoundError: if model_path does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at '{model_path}'. "
            "Run src/train.py first to train and save the model."
        )
    print(f"Loading model from {model_path} ...")
    try:
        import keras
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass
    
    try:
        return keras_load_model(model_path, safe_mode=False, compile=False)
    except (TypeError, ValueError):
        return keras_load_model(model_path, compile=False)


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_path, img_size=IMG_SIZE):
    """
    Load and preprocess a single MRI image for inference.

    Args:
        img_path: Path to the image file.
        img_size: (height, width) tuple.

    Returns:
        ndarray of shape (1, H, W, 3), dtype float32, values in [0, 1].
    """
    img = load_img(img_path, target_size=img_size, color_mode="rgb")
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Single image prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(img_path, model=None, threshold=None):
    """
    Run inference on a single MRI image.
    """
    if model is None:
        model = load_model()

    arr  = preprocess_image(img_path)
    # Wrap in list to avoid Keras 3 functional input warning
    probs = model.predict([arr], verbose=0)[0]
    pred_idx = np.argmax(probs)
    label    = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    is_tumor = label != "notumor"

    result = dict(
        label=label,
        confidence=round(confidence * 100, 2),
        probability=round(confidence * 100, 2),
        is_tumor=is_tumor,
        class_idx=int(pred_idx)
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(img_paths, model=None, threshold=THRESHOLD):
    """
    Run inference on a list of MRI images.

    Args:
        img_paths: List of image file paths.
        model:     Loaded Keras model (loads from disk if None).
        threshold: Decision threshold.

    Returns:
        List of result dicts (same format as predict_image).
    """
    if model is None:
        model = load_model()

    results = []
    for path in img_paths:
        try:
            results.append(predict_image(path, model, threshold))
        except Exception as exc:
            results.append({"error": str(exc), "path": path})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    import tensorflow as tf
    try:
        # Search for MobileNetV2 layer dynamically in case names differ
        base_model = None
        for layer in model.layers:
            if "mobilenetv2" in layer.name.lower():
                base_model = layer
                break
        
        if base_model is None:
            return None
            
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
    except Exception:
        return None

    inner_grad_model = tf.keras.models.Model(
        [base_model.inputs], [last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        x = model.get_layer("rescaling")(img_array)
        tape.watch(x)
        
        last_conv_output, base_output = inner_grad_model(x)
        
        x = model.get_layer("avg_pool")(base_output)
        x = model.get_layer("fc_128")(x)
        x = model.get_layer("batch_normalization")(x)
        
        # Use the predicted class channel for Grad-CAM
        preds = model.get_layer("output")(x)
        pred_idx = tf.argmax(preds[0])
        class_channel = preds[:, pred_idx]

    grads = tape.gradient(class_channel, last_conv_output)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat > 0:
        heatmap /= max_heat
        
    return heatmap.numpy()

def visualise_prediction(img_path, result, model=None):
    """
    Display the MRI image with the prediction overlaid and Grad-CAM Heatmap.

    Args:
        img_path: Path to the image.
        result:   Dict returned by predict_image.
        model:    Loaded model to generate Grad-CAM (optional).
    """
    img = load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    arr = img_to_array(img).astype(np.uint8)

    color = "red" if result["is_tumor"] else "green"
    title = (
        f"{result['label']}  |  "
        f"Confidence: {result['confidence']:.1f}%  |  "
        f"P(tumor): {result['probability']:.1f}%"
    )

    heatmap = None
    if model is not None:
        img_array = preprocess_image(img_path)
        heatmap = make_gradcam_heatmap(img_array, model)

    if heatmap is not None:
        import cv2
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(arr)
        axes[0].set_title(title, color=color, fontsize=12, fontweight="bold")
        axes[0].axis("off")

        heatmap_resized = cv2.resize(heatmap, (arr.shape[1], arr.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.addWeighted(arr, 0.6, heatmap_colored, 0.4, 0)

        axes[1].imshow(superimposed_img)
        axes[1].set_title("Explainable AI (Grad-CAM Heatmap)", fontsize=12, fontweight="bold")
        axes[1].axis("off")
    else:
        plt.figure(figsize=(5, 5))
        plt.imshow(arr)
        plt.title(title, color=color, fontsize=12, fontweight="bold")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Brain MRI Tumor Detection — inference script"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the MRI image file (.jpg / .jpeg / .png)"
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Decision threshold (default: {THRESHOLD})"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to saved model (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--visualise", action="store_true",
        help="Display the image with prediction overlay"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    loaded_model = load_model(args.model)
    result       = predict_image(args.image, loaded_model, args.threshold)

    print("\n" + "=" * 45)
    print(f"  Image      : {os.path.basename(args.image)}")
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']:.1f}%")
    print(f"  P(tumor)   : {result['probability']:.1f}%")
    print("=" * 45)

    if args.visualise:
        visualise_prediction(args.image, result, loaded_model)
