"""
diagnose_per_class.py
=====================
Safe Pharmacy — Per-class accuracy diagnostic.

Uses a lightweight frozen MobileNetV2 (no fine-tuning at all)
to get honest per-class accuracy without overfitting.
Frozen backbone = zero risk of memorising training data.

Outputs -> saved_models/diagnostics/
    per_class_report.txt
    per_class_report.json
    per_class_accuracy.png
    confusion_matrix.png
    baseline.keras
    baseline.h5
    baseline.weights.h5
"""

import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
DIAG_DIR      = BASE_DIR / "saved_models" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = (224, 224)
IMG_SHAPE   = (224, 224, 3)
BATCH_SIZE  = 16
EPOCHS      = 30
NUM_CLASSES = 19

# ---------------------------------------------------------------------------
# DATA — strong augmentation to fight overfitting
# ---------------------------------------------------------------------------

def build_generators():
    # Heavy runtime augmentation on train
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train = train_datagen.flow_from_directory(
        PROCESSED_DIR / "train",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True, seed=42,
    )
    val = eval_datagen.flow_from_directory(
        PROCESSED_DIR / "val",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    test = eval_datagen.flow_from_directory(
        PROCESSED_DIR / "test",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    return train, val, test


def get_class_weights(train_gen):
    y = train_gen.classes
    w = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return dict(enumerate(w))

# ---------------------------------------------------------------------------
# MODEL — fully frozen backbone, tiny head
# ---------------------------------------------------------------------------

def build_model(num_classes):
    """
    Fully frozen MobileNetV2 + small head.
    Frozen backbone means no overfitting from the feature extractor.
    Small head (128 units only) means less capacity to memorise.
    """
    base = MobileNetV2(input_shape=IMG_SHAPE, include_top=False,
                       weights="imagenet")
    base.trainable = False      # FULLY FROZEN — no overfitting from backbone

    inp = layers.Input(shape=IMG_SHAPE, name="image_input")
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(128, activation="relu",
                       kernel_regularizer=regularizers.l2(1e-3))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax",
                       name="class_output")(x)

    return models.Model(inp, out, name="baseline_frozen")


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(model, train_gen, val_gen, class_weights):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(DIAG_DIR / "baseline_best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    print("\n[Training frozen baseline ...]")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1,
    )
    return history

# ---------------------------------------------------------------------------
# PER-CLASS ANALYSIS
# ---------------------------------------------------------------------------

def analyse(model, test_gen, val_gen, history):
    classes = list(test_gen.class_indices.keys())

    # Test predictions
    test_gen.reset()
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    test_gen.reset()
    y_proba = model.predict(test_gen, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)
    y_true  = test_gen.classes

    # Val predictions
    val_gen.reset()
    _, val_acc = model.evaluate(val_gen, verbose=0)
    val_gen.reset()
    y_val_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
    y_val_true = val_gen.classes

    # Per-class metrics from test
    cm           = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    per_cls_acc  = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    prec_t, rec_t, f1_t, sup_t = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)),
        average=None, zero_division=0,
    )
    # Per-class metrics from val
    prec_v, rec_v, f1_v, sup_v = precision_recall_fscore_support(
        y_val_true, y_val_pred, labels=range(len(classes)),
        average=None, zero_division=0,
    )

    # Train accuracy per class from last epoch history
    train_acc_overall = history.history["accuracy"][-1]
    val_acc_overall   = history.history["val_accuracy"][-1]
    overall_gap       = train_acc_overall - val_acc_overall

    results = {}
    for i, cls in enumerate(classes):
        train_dir = PROCESSED_DIR / "train" / cls
        n_train   = sum(1 for f in train_dir.iterdir()
                        if f.suffix == ".jpg") if train_dir.exists() else 0

        gap_f1       = float(f1_t[i]) - float(f1_v[i])
        overfit_flag = gap_f1 > 0.20
        low_acc_flag = float(per_cls_acc[i]) < 0.40

        results[cls] = {
            "train_images":  n_train,
            "test_images":   int(sup_t[i]),
            "val_images":    int(sup_v[i]),
            "test_accuracy": round(float(per_cls_acc[i]) * 100, 1),
            "test_precision":round(float(prec_t[i]) * 100, 1),
            "test_recall":   round(float(rec_t[i]) * 100, 1),
            "test_f1":       round(float(f1_t[i]) * 100, 1),
            "val_f1":        round(float(f1_v[i]) * 100, 1),
            "f1_gap":        round(gap_f1 * 100, 1),
            "overfit_flag":  overfit_flag,
            "low_acc_flag":  low_acc_flag,
        }

    summary = {
        "overall_train_acc": round(train_acc_overall * 100, 2),
        "overall_val_acc":   round(val_acc_overall   * 100, 2),
        "overall_test_acc":  round(test_acc          * 100, 2),
        "overall_gap":       round(overall_gap       * 100, 2),
        "classes":           results,
    }
    return summary, cm, classes

# ---------------------------------------------------------------------------
# REPORTS
# ---------------------------------------------------------------------------

def write_report(summary, txt_path):
    r       = summary
    classes = r["classes"]

    lines = []
    lines.append("=" * 75)
    lines.append("  Safe Pharmacy — Per-Class Accuracy Report (Frozen Baseline)")
    lines.append("=" * 75)
    lines.append(f"\n  Overall Train Accuracy : {r['overall_train_acc']}%")
    lines.append(f"  Overall Val  Accuracy  : {r['overall_val_acc']}%")
    lines.append(f"  Overall Test Accuracy  : {r['overall_test_acc']}%")
    lines.append(f"  Train/Val Gap          : {r['overall_gap']}%")
    lines.append("")
    lines.append("  FLAGS:")
    lines.append("    [OVERFIT] = val_f1 is 20%+ lower than test_f1")
    lines.append("    [LOW ACC] = per-class test accuracy < 40%")
    lines.append("")
    lines.append("-" * 75)
    lines.append(
        f"  {'Class':<32} {'Train':>6} {'Acc%':>6} "
        f"{'Prec%':>6} {'Rec%':>6} {'F1%':>5} {'valF1':>6} {'Gap':>5}  Flags"
    )
    lines.append("-" * 75)

    for cls, d in sorted(classes.items(), key=lambda x: x[1]["test_accuracy"]):
        flags = ""
        if d["overfit_flag"]: flags += "[OVERFIT] "
        if d["low_acc_flag"]: flags += "[LOW ACC]"
        lines.append(
            f"  {cls:<32} {d['train_images']:>6} "
            f"{d['test_accuracy']:>5.1f}% "
            f"{d['test_precision']:>5.1f}% "
            f"{d['test_recall']:>5.1f}% "
            f"{d['test_f1']:>4.1f}% "
            f"{d['val_f1']:>5.1f}% "
            f"{d['f1_gap']:>+5.1f}%  {flags}"
        )

    lines.append("\n" + "=" * 75)
    lines.append("  RECOMMENDATIONS")
    lines.append("=" * 75)

    very_low = [(c, d) for c, d in classes.items() if d["test_accuracy"] < 20]
    low_acc  = [(c, d) for c, d in classes.items() if 20 <= d["test_accuracy"] < 40]
    overfit  = [(c, d) for c, d in classes.items() if d["overfit_flag"]]

    if very_low:
        lines.append("\n  CONSIDER DROPPING (acc < 20%) — too few or too similar images:")
        for c, d in very_low:
            lines.append(f"    - {c}  (train={d['train_images']}, acc={d['test_accuracy']}%)")

    if low_acc:
        lines.append("\n  LOW ACCURACY classes (20-40%) — need more images or merge:")
        for c, d in low_acc:
            lines.append(f"    - {c}  (train={d['train_images']}, acc={d['test_accuracy']}%)")

    if overfit:
        lines.append("\n  OVERFIT classes — val_f1 much lower than test_f1:")
        for c, d in overfit:
            lines.append(f"    - {c}  (gap={d['f1_gap']:+.1f}%)")

    if not very_low and not low_acc and not overfit:
        lines.append("\n  All classes look reasonable. Proceed with multimodel training.")

    text = "\n".join(lines)
    txt_path.write_text(text, encoding="utf-8")
    print(text)


def plot_per_class(summary, classes_order):
    classes = summary["classes"]
    sorted_items = sorted(classes.items(), key=lambda x: x[1]["test_accuracy"])
    names   = [c.replace("_", "\n") for c, _ in sorted_items]
    test_f1 = [d["test_f1"]         for _, d in sorted_items]
    val_f1  = [d["val_f1"]          for _, d in sorted_items]
    acc     = [d["test_accuracy"]   for _, d in sorted_items]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Per-Class Diagnostic — Frozen MobileNetV2 Baseline", fontsize=13)

    # Chart 1: test accuracy bar
    colors = ["#A32D2D" if a < 20 else "#BA7517" if a < 40 else "#0F6E56"
              for a in acc]
    axes[0].barh(range(len(names)), acc, color=colors)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].axvline(40, color="#BA7517", linestyle="--", lw=1.2, label="40% threshold")
    axes[0].axvline(20, color="#A32D2D", linestyle="--", lw=1.2, label="20% drop threshold")
    axes[0].set_xlabel("Test Accuracy (%)")
    axes[0].set_title("Per-Class Test Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.3)
    for i, v in enumerate(acc):
        axes[0].text(v + 0.3, i, f"{v:.0f}%", va="center", fontsize=7)

    # Chart 2: test_f1 vs val_f1 — shows overfitting gap per class
    x     = np.arange(len(names))
    width = 0.35
    axes[1].barh(x - width/2, test_f1, width, label="Test F1",  color="#534AB7", alpha=0.85)
    axes[1].barh(x + width/2, val_f1,  width, label="Val  F1",  color="#E07B3F", alpha=0.85)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].set_xlabel("F1 Score (%)")
    axes[1].set_title("Test F1 vs Val F1 per Class\n(large gap = overfitting risk)")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    p = DIAG_DIR / "per_class_accuracy.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Per-class chart saved: {p}")


def plot_confusion(cm, classes):
    fig, ax = plt.subplots(figsize=(14, 12))
    short   = [c[:14] for c in classes]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax.set_title("Confusion Matrix — Frozen Baseline Diagnostic")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    p = DIAG_DIR / "confusion_matrix.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {p}")

# ---------------------------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------------------------

def save_model(model):
    model.save(DIAG_DIR / "baseline.keras")
    model.save(DIAG_DIR / "baseline.h5")
    model.save_weights(str(DIAG_DIR / "baseline.weights.h5"))
    print("  Saved: baseline.keras / baseline.h5 / baseline.weights.h5")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"  GPU: {gpus[0].name}")
    else:
        print("  No GPU — CPU only (slower)")

    print("=" * 60)
    print("  Safe Pharmacy — diagnose_per_class.py")
    print("=" * 60)

    train_gen, val_gen, test_gen = build_generators()
    class_weights = get_class_weights(train_gen)

    model   = build_model(NUM_CLASSES)
    history = train_model(model, train_gen, val_gen, class_weights)

    save_model(model)

    print("\n[Analysing per-class accuracy ...]")
    summary, cm, classes = analyse(model, test_gen, val_gen, history)

    # Save JSON
    json_p = DIAG_DIR / "per_class_report.json"
    with open(json_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON saved: {json_p}")

    write_report(summary, DIAG_DIR / "per_class_report.txt")
    plot_per_class(summary, classes)
    plot_confusion(cm, classes)

    # Quick console summary
    cls_data = summary["classes"]
    print("\n" + "=" * 60)
    print("  QUICK SUMMARY")
    print("=" * 60)
    print(f"  {'Class':<35} {'Acc%':>6}  {'Gap':>6}  Status")
    print("-" * 60)
    for cls, d in sorted(cls_data.items(), key=lambda x: x[1]["test_accuracy"]):
        status = ""
        if d["test_accuracy"] < 20:  status = "CONSIDER DROP"
        elif d["low_acc_flag"]:      status = "LOW - needs more data"
        elif d["overfit_flag"]:      status = "OVERFIT"
        else:                        status = "OK"
        print(f"  {cls:<35} {d['test_accuracy']:>5.1f}%  "
              f"{d['f1_gap']:>+5.1f}%  {status}")

    print(f"\n  Overall: train={summary['overall_train_acc']}%  "
          f"val={summary['overall_val_acc']}%  "
          f"test={summary['overall_test_acc']}%")
    print(f"\n  Full report: saved_models/diagnostics/per_class_report.txt")
    print("\n  Done.")