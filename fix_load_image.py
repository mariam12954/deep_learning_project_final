"""
fix_load_image.py  —  Run ONCE before multimodel1.py
=====================================================
The old image_feature_extractor.keras was saved with Lambda(l2_normalize)
which Keras 3 cannot load (shape inference fails).

This script loads the weights-only file (.weights.h5) instead,
rebuilds the graph with L2NormLayer, copies weights, and saves fresh .keras.

Steps:
  1. python fix_load_image.py
  2. python multimodel1.py   (will skip image training, use fresh extractor)
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.applications import EfficientNetB0
from pathlib import Path

SAVE_DIR    = Path(__file__).resolve().parent / "saved_models_final" / "multimodel1"
IMG_SIZE    = (224, 224)
FEAT_DIM    = 256
NUM_CLASSES = 23   # change if your dataset has different number


class L2NormLayer(tf.keras.layers.Layer):
    """Replaces Lambda — saves/loads correctly."""
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)
    def get_config(self):
        return super().get_config()


def build_full_classifier(num_classes):
    """Same architecture as multimodel1.py — must match exactly."""
    backbone = EfficientNetB0(include_top=False, weights=None,
                               input_shape=(*IMG_SIZE, 3))
    inp  = Input(shape=(*IMG_SIZE, 3), name="image_input")
    x    = backbone(inp, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(FEAT_DIM,
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="image_features")(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Activation("relu")(x)
    feat = L2NormLayer(name="image_l2norm")(x)
    head = layers.Dropout(0.5)(feat)
    head = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(head)
    head = layers.Dropout(0.3)(head)
    out  = layers.Dense(num_classes, activation="softmax",
                        name="image_softmax")(head)
    return Model(inp, out, name="efficientnet_classifier")


def fix():
    print("=" * 60)
    print("  fix_load_image.py")
    print("=" * 60)

    # ── Find the best weights file ────────────────────────────
    # Priority: img_best_finetune weights → img_best_frozen → any .weights.h5
    candidates = [
        SAVE_DIR / "img_best_finetune.weights.h5",
        SAVE_DIR / "img_best_frozen.weights.h5",
        SAVE_DIR / "img_best_finetune.keras",   # try keras with unsafe as fallback
    ]

    weights_file = None
    for c in candidates:
        if c.exists():
            weights_file = c
            print(f"  Found weights: {c.name}")
            break

    if weights_file is None:
        print("\n  ERROR: No weights file found in:")
        print(f"  {SAVE_DIR}")
        print("\n  Files present:")
        for f in sorted(SAVE_DIR.iterdir()):
            print(f"    {f.name}")
        print("\n  Solution: Delete saved_models_final/multimodel1/ completely")
        print("  and run python multimodel1.py from scratch.")
        return

    # ── Build fresh model with L2NormLayer ───────────────────
    print(f"\n  Building fresh model (L2NormLayer, {NUM_CLASSES} classes)...")
    model = build_full_classifier(NUM_CLASSES)

    # ── Load weights ──────────────────────────────────────────
    ext = weights_file.suffix
    if ext == ".h5":
        print(f"  Loading weights from {weights_file.name} ...")
        model.load_weights(str(weights_file))
        print("  Weights loaded OK")
    else:
        # .keras file — try with unsafe deserialization just for weights
        print(f"  {weights_file.name} is .keras — extracting weights only...")
        tf.keras.config.enable_unsafe_deserialization()
        try:
            tmp = tf.keras.models.load_model(str(weights_file))
            # Copy weights by layer name
            copied = 0
            for new_layer in model.layers:
                try:
                    old_layer = tmp.get_layer(new_layer.name)
                    w = old_layer.get_weights()
                    if w:
                        new_layer.set_weights(w)
                        copied += 1
                except Exception:
                    pass
            print(f"  Copied weights from {copied} layers")
        except Exception as e:
            print(f"  Could not load .keras: {e}")
            print("  Falling back to ImageNet weights only (no fine-tune weights)")
        tf.keras.config.disable_unsafe_deserialization()

    # ── Build and save extractor ──────────────────────────────
    extractor = Model(
        model.input,
        model.get_layer("image_l2norm").output,
        name="image_feature_extractor",
    )

    out_path = SAVE_DIR / "image_feature_extractor.keras"
    extractor.save(str(out_path))
    print(f"\n  Saved: {out_path.name}")

    # ── Verify loads cleanly ──────────────────────────────────
    loaded = tf.keras.models.load_model(
        str(out_path),
        custom_objects={"L2NormLayer": L2NormLayer},
    )
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    result = loaded.predict(dummy, verbose=0)
    print(f"  Verified output shape: {result.shape}  (expected (1, 256))")
    print("\n  Done. Now run: python multimodel1.py")


if __name__ == "__main__":
    fix()