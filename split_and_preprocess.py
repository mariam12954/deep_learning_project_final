"""
split_and_preprocess_final.py  —  Safe Pharmacy
================================================
WHY THE OLD VERSION CAUSED OVERFITTING:
  1. Augmented images were saved to disk THEN the online generator
     augmented them AGAIN → double augmentation on train, zero on val
     → model sees very different distributions → fake low val acc
  2. rescale=1/255 was mixed with preprocess_input in places
  3. No preprocess_input on val/test → completely different normalisation

FIXES HERE:
  • Offline aug saves PLAIN resized JPEGs (PIL only, no preprocess_input)
  • preprocess_input applied ONLY at model.fit() time via generator
  • Val/Test: resize only — zero augmentation — zero preprocess at disk level
  • Augmentation is mild (rotation ±10°, no shear, no vertical flip)
  • get_data_generators() is exported for all model files to import

Split: 70 / 20 / 10
Target: 350 images per class (offline aug)
"""

import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dataset"

IMG_SIZE         = (224, 224)
SAVE_FORMAT      = "JPEG"
SAVE_QUALITY     = 95
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.20
AUGMENT_TRAIN    = False #اتجنب الاوفرفيتنج
TARGET_PER_CLASS = 180
MAX_COPIES       = 4
IMAGE_EXTS       = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}
RANDOM_SEED      = 42
BATCH_SIZE       = 16


# ── Augmentation (PIL only — no preprocess_input) ──────────────
def random_augment(img: Image.Image) -> Image.Image:
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.5:
        img = img.rotate(random.uniform(-10, 10),
                         resample=Image.BILINEAR, expand=False)
    if random.random() > 0.3:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.82, 1.18))
    if random.random() > 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    if random.random() > 0.5:
        img = ImageEnhance.Saturation(img).enhance(random.uniform(0.85, 1.15))
    if random.random() > 0.5:
        w, h = img.size
        crop = random.uniform(0.90, 1.0)
        left = int(w * (1 - crop) / 2)
        top  = int(h * (1 - crop) / 2)
        img  = img.crop((left, top, w - left, h - top))
        img  = img.resize((w, h), Image.BILINEAR)
    # Random erasing 8% chance only
    if random.random() > 0.92:
        w, h = img.size
        rw = int(w * random.uniform(0.05, 0.12))
        rh = int(h * random.uniform(0.05, 0.12))
        rx = random.randint(0, w - rw)
        ry = random.randint(0, h - rh)
        ImageDraw.Draw(img).rectangle([rx, ry, rx+rw, ry+rh], fill=(0,0,0))
    return img


def save_image(img: Image.Image, path: Path):
    img.save(path, format=SAVE_FORMAT, quality=SAVE_QUALITY)


def open_and_resize(src: Path) -> Image.Image:
    with Image.open(src) as im:
        return im.convert("RGB").resize(IMG_SIZE, Image.LANCZOS).copy()


def compute_aug_copies(n_train: int) -> int:
    if n_train == 0 or n_train >= TARGET_PER_CLASS:
        return 0
    needed = TARGET_PER_CLASS - n_train
    return min((needed + n_train - 1) // n_train, MAX_COPIES)


# ── Step 1: Split ──────────────────────────────────────────────
def split_images():
    random.seed(RANDOM_SEED)
    images_dir = OUTPUT_DIR / "images"
    split_base = OUTPUT_DIR / "split"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Not found: {images_dir}\nRun organize_v3.py first."
        )
    for s in ("train", "val", "test"):
        (split_base / s).mkdir(parents=True, exist_ok=True)

    totals, per_class = {"train": 0, "val": 0, "test": 0}, {}
    for cls_dir in sorted(d for d in images_dir.iterdir() if d.is_dir()):
        imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        random.shuffle(imgs)
        n       = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        splits  = {
            "train": imgs[:n_train],
            "val":   imgs[n_train: n_train + n_val],
            "test":  imgs[n_train + n_val:],
        }
        per_class[cls_dir.name] = {}
        for sname, files in splits.items():
            dest = split_base / sname / cls_dir.name
            dest.mkdir(exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
            totals[sname]                  += len(files)
            per_class[cls_dir.name][sname]  = len(files)
    return totals, per_class


# ── Step 2: Offline preprocess (PIL only, no preprocess_input) ─
def preprocess_split(split_name: str, augment: bool = False) -> int:
    """
    Saves plain resized JPEGs to disk.
    NO preprocess_input here — that happens inside the generator at train time.
    Train: resize + mild PIL augmentation to reach TARGET_PER_CLASS.
    Val/Test: resize only.
    """
    src_base = OUTPUT_DIR / "split"     / split_name
    dst_base = OUTPUT_DIR / "processed" / split_name
    if not src_base.exists():
        print(f"  Not found: {src_base} (skipping)")
        return 0

    total = 0
    for cls_dir in sorted(d for d in src_base.iterdir() if d.is_dir()):
        dst_cls = dst_base / cls_dir.name
        dst_cls.mkdir(parents=True, exist_ok=True)
        images  = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        copies  = compute_aug_copies(len(images)) if augment else 0
        desc    = f"  {split_name}/{cls_dir.name}" + (f" +{copies}x" if copies else "")

        for img_path in tqdm(images, desc=desc, leave=False):
            try:
                img = open_and_resize(img_path)
                save_image(img, dst_cls / (img_path.stem + ".jpg"))
                total += 1
            except Exception as e:
                print(f"\n  Skip {img_path.name}: {e}")
                continue
            for i in range(copies):
                try:
                    aug = random_augment(open_and_resize(img_path))
                    save_image(aug, dst_cls / f"{img_path.stem}_aug{i+1}.jpg")
                    total += 1
                except Exception as e:
                    print(f"\n  Aug error {img_path.name}: {e}")

    print(f"   {split_name}: {total} images → {dst_base}")
    return total


# ── Step 3: Generator factory (imported by all model files) ────
def get_generators(backbone: str = "efficientnet"):
    """
    Returns (train_gen, val_gen, test_gen) ready for model.fit().

    backbone: "efficientnet" | "mobilenet"
      Each uses the correct preprocess_input for that backbone.

    CRITICAL: preprocess_input is applied ONLY here (online, in RAM).
              The disk images are plain JPEGs — no preprocessing baked in.
              Val/Test: preprocess_input only, ZERO augmentation.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if backbone == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif backbone == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    base = OUTPUT_DIR / "processed"

    # Online augmentation (mild, in RAM) + correct preprocess_input
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=8,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        str(base / "train"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
        shuffle=True, seed=RANDOM_SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        str(base / "val"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        str(base / "test"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ── Summary ────────────────────────────────────────────────────
def print_split_summary(totals, per_class):
    total_all = sum(totals.values())
    print("\n" + "=" * 70)
    print("  Split Summary  (70 / 20 / 10)")
    print("=" * 70)
    print(f"  Total : {total_all}")
    for s in ("train", "val", "test"):
        print(f"  {s:<6}: {totals[s]:>5}  ({totals[s]/total_all*100:.1f}%)")
    print("\n" + "-" * 70)
    print(f"  {'Class':<42} {'Train':>6} {'Val':>5} {'Test':>5} {'Aug':>8}")
    print("-" * 70)
    for cls, c in sorted(per_class.items(), key=lambda x: -x[1]["train"]):
        cp = compute_aug_copies(c["train"]) if AUGMENT_TRAIN else 0
        print(f"  {cls:<42} {c['train']:>6} {c['val']:>5} {c['test']:>5}   "
              f"{'—' if cp == 0 else f'+{cp}x'}")
    print("-" * 70)


def print_processed_summary():
    base = OUTPUT_DIR / "processed"
    print("\n" + "=" * 70)
    print("  Processed Dataset Summary (after offline augmentation)")
    print("=" * 70)
    for s in ("train", "val", "test"):
        d = base / s
        if not d.exists():
            continue
        total  = sum(1 for f in d.rglob("*") if f.suffix == ".jpg")
        n_cls  = sum(1 for x in d.iterdir() if x.is_dir())
        counts = sorted(
            [(x.name, sum(1 for f in x.iterdir() if f.suffix == ".jpg"))
             for x in d.iterdir() if x.is_dir()],
            key=lambda x: x[1],
        )
        if s == "train" and counts:
            mn, mx = counts[0][1], counts[-1][1]
            ratio  = mx / mn if mn > 0 else float("inf")
            print(f"  {s:<6}: {total:>5} images | {n_cls} classes "
                  f"| min {mn} / max {mx} | imbalance {ratio:.1f}x")
            for name, cnt in sorted(counts, key=lambda x: -x[1]):
                print(f"         {name:<42} {cnt:>5}")
        else:
            print(f"  {s:<6}: {total:>5} images | {n_cls} classes")
    print("=" * 70)
    print(f"\n  Path: {base}")
    print("  Next → python multimodel1.py  then  multimodel2.py")


if __name__ == "__main__":
    print("=" * 70)
    print("  split_and_preprocess_final.py")
    print("  Split: 70/20/10 | PIL offline aug | preprocess_input online only")
    print("=" * 70)
    print(f"  TARGET_PER_CLASS={TARGET_PER_CLASS}  MAX_COPIES={MAX_COPIES}")

    print("\n[1/4] Splitting 70 / 20 / 10 ...")
    totals, per_class = split_images()
    print_split_summary(totals, per_class)

    print("\n[2/4] TRAIN — resize + PIL aug (no preprocess_input) ...")
    preprocess_split("train", augment=False) #تجنب الاوفرفيتنج  

    print("\n[3/4] VAL — resize only ...")
    preprocess_split("val", augment=False)

    print("\n[4/4] TEST — resize only ...")
    preprocess_split("test", augment=False)

    print_processed_summary()
    print("\n  Done.")