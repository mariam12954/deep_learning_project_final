"""
data_quality_fix.py  —  Safe Pharmacy
=======================================
Automated data quality pipeline BEFORE split_and_preprocess.

Fixes IN ORDER:
    1. Remove blurry images        (Laplacian variance < threshold)
    2. Remove dark images          (mean brightness < threshold)
    3. Remove duplicates           (perceptual hash)
    4. Center crop to subject      (80% center crop removes background noise)
    5. Enhance contrast + sharpness (mild — keeps text readable)
    6. Report mislabel candidates  (images too dissimilar from class centroid)

Run ONCE on dataset/images/ BEFORE split_and_preprocess.py

Usage:
    python data_quality_fix.py            # dry run — shows what would be removed
    python data_quality_fix.py --apply    # actually removes/fixes images
    python data_quality_fix.py --apply --report-only  # fixes + saves report only

Output:
    dataset/images/           cleaned images (in-place)
    dataset/text/quality_report.txt
    dataset/text/mislabel_candidates.txt
"""

import argparse
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — adjust thresholds if too aggressive or too lenient
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent
IMAGES_DIR    = BASE_DIR / "dataset" / "images"
REPORT_DIR    = BASE_DIR / "dataset" / "text"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE      = (224, 224)
IMAGE_EXTS    = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}

# Quality thresholds
BLUR_THRESHOLD       = 80.0    # Laplacian variance — below this = blurry
DARK_THRESHOLD       = 35.0    # mean pixel brightness 0-255 — below = too dark
BRIGHT_THRESHOLD     = 245.0   # mean brightness — above = overexposed

# Augmentation reduction
TARGET_PER_CLASS_NEW = 180     # reduced from 350 — less aug = less overfitting
MAX_COPIES_NEW       = 4       # reduced from 12

# Center crop ratio
CROP_RATIO           = 0.82    # keep central 82% of image

# Enhancement (mild)
CONTRAST_FACTOR      = 1.15    # 1.0 = no change
SHARPNESS_FACTOR     = 1.20

# Mislabel detection
MISLABEL_PERCENTILE  = 15      # flag images in bottom 15% similarity to class

# ---------------------------------------------------------------------------
# STEP 1 — BLUR DETECTION
# ---------------------------------------------------------------------------

def laplacian_variance(img_array: np.ndarray) -> float:
    """Higher = sharper. Blurry images have low variance."""
    gray = np.mean(img_array, axis=2)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Manual convolution on center crop (faster than scipy)
    h, w = gray.shape
    pad  = np.pad(gray, 1, mode="edge")
    lap  = (pad[:-2, 1:-1] + pad[2:, 1:-1] +
            pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * gray)
    return float(np.var(lap))


def is_blurry(img_array: np.ndarray) -> bool:
    return laplacian_variance(img_array) < BLUR_THRESHOLD


# ---------------------------------------------------------------------------
# STEP 2 — BRIGHTNESS / EXPOSURE CHECK
# ---------------------------------------------------------------------------

def is_too_dark(img_array: np.ndarray) -> bool:
    return float(np.mean(img_array)) < DARK_THRESHOLD


def is_overexposed(img_array: np.ndarray) -> bool:
    return float(np.mean(img_array)) > BRIGHT_THRESHOLD


# ---------------------------------------------------------------------------
# STEP 3 — DUPLICATE DETECTION (perceptual hash)
# ---------------------------------------------------------------------------

def phash(img: Image.Image, hash_size: int = 16) -> str:
    """Simple average-hash for near-duplicate detection."""
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr   = np.array(small, dtype=np.float32)
    mean  = arr.mean()
    bits  = (arr > mean).flatten()
    return "".join(str(int(b)) for b in bits)


def hamming(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def find_duplicates(image_files: list[Path],
                    threshold: int = 8) -> set[Path]:
    """Return set of duplicate paths (keeps one, marks rest for removal)."""
    hashes   = {}
    to_remove = set()

    for fp in image_files:
        try:
            with Image.open(fp) as im:
                h = phash(im.convert("RGB"))
        except Exception:
            continue

        matched = False
        for existing_hash, existing_path in hashes.items():
            if hamming(h, existing_hash) <= threshold:
                to_remove.add(fp)
                matched = True
                break
        if not matched:
            hashes[h] = fp

    return to_remove


# ---------------------------------------------------------------------------
# STEP 4 — CENTER CROP
# ---------------------------------------------------------------------------

def center_crop(img: Image.Image, ratio: float = CROP_RATIO) -> Image.Image:
    w, h   = img.size
    new_w  = int(w * ratio)
    new_h  = int(h * ratio)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


# ---------------------------------------------------------------------------
# STEP 5 — ENHANCE (contrast + sharpness)
# ---------------------------------------------------------------------------

def enhance_image(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(CONTRAST_FACTOR)
    img = ImageEnhance.Sharpness(img).enhance(SHARPNESS_FACTOR)
    return img


# ---------------------------------------------------------------------------
# STEP 6 — MISLABEL CANDIDATES
# ---------------------------------------------------------------------------

def class_centroid(image_files: list[Path]) -> np.ndarray:
    """Compute mean feature vector (flattened resized image) for a class."""
    vectors = []
    for fp in image_files:
        try:
            with Image.open(fp) as im:
                arr = np.array(im.convert("RGB").resize((32, 32)),
                               dtype=np.float32) / 255.0
                vectors.append(arr.flatten())
        except Exception:
            continue
    if not vectors:
        return np.zeros(32 * 32 * 3)
    return np.mean(vectors, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_mislabel_candidates(cls_dir: Path,
                             image_files: list[Path]) -> list[tuple]:
    """
    Images with cosine similarity to class centroid below 15th percentile
    are flagged as potential mislabels.
    Returns list of (path, similarity_score).
    """
    centroid = class_centroid(image_files)
    sims     = []

    for fp in image_files:
        try:
            with Image.open(fp) as im:
                arr = np.array(im.convert("RGB").resize((32, 32)),
                               dtype=np.float32) / 255.0
                vec = arr.flatten()
                sims.append((fp, cosine_similarity(centroid, vec)))
        except Exception:
            sims.append((fp, 0.0))

    if not sims:
        return []

    scores     = [s for _, s in sims]
    threshold  = float(np.percentile(scores, MISLABEL_PERCENTILE))
    candidates = [(fp, round(sim, 4))
                  for fp, sim in sims if sim < threshold]
    return sorted(candidates, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(apply: bool = False, report_only: bool = False):
    if not IMAGES_DIR.exists():
        print(f"  ERROR: {IMAGES_DIR} not found. Run organize.py first.")
        return

    mode = "APPLY" if apply else "DRY RUN"
    print("=" * 65)
    print(f"  Safe Pharmacy — data_quality_fix.py  [{mode}]")
    print("=" * 65)
    if not apply:
        print("  (Pass --apply to actually remove/fix images)\n")

    total_removed   = 0
    total_enhanced  = 0
    report_lines    = []
    mislabel_lines  = []
    class_summary   = {}

    class_dirs = sorted(d for d in IMAGES_DIR.iterdir() if d.is_dir())

    for cls_dir in class_dirs:
        images = [f for f in cls_dir.iterdir()
                  if f.suffix.lower() in IMAGE_EXTS]
        if not images:
            continue

        removed_blur    = []
        removed_dark    = []
        removed_dup     = []
        enhanced_list   = []

        # ── STEP 3: duplicates first (cheapest) ──────────────────────
        dups = find_duplicates(images)

        # ── PER-IMAGE quality checks ──────────────────────────────────
        for fp in tqdm(images, desc=f"  {cls_dir.name}", leave=False):
            if fp in dups:
                removed_dup.append(fp)
                if apply:
                    fp.unlink(missing_ok=True)
                continue

            try:
                with Image.open(fp) as im:
                    img = im.convert("RGB")
                    arr = np.array(img, dtype=np.float32)
            except Exception as e:
                print(f"\n    Skip {fp.name}: {e}")
                continue

            # Quality filters
            if is_blurry(arr):
                removed_blur.append(fp)
                if apply:
                    fp.unlink(missing_ok=True)
                continue

            if is_too_dark(arr) or is_overexposed(arr):
                removed_dark.append(fp)
                if apply:
                    fp.unlink(missing_ok=True)
                continue

            # Center crop + enhance + save (only if apply)
            if apply:
                try:
                    img_cropped  = center_crop(img)
                    img_enhanced = enhance_image(img_cropped)
                    img_final    = img_enhanced.resize(IMG_SIZE, Image.LANCZOS)
                    img_final.save(fp, format="JPEG", quality=95)
                    enhanced_list.append(fp)
                except Exception as e:
                    print(f"\n    Enhance error {fp.name}: {e}")

        # ── STEP 6: mislabel candidates ───────────────────────────────
        remaining = [f for f in images
                     if f not in removed_blur
                     and f not in removed_dark
                     and f not in removed_dup]

        mislabels = find_mislabel_candidates(cls_dir, remaining)

        # Counts
        n_removed = len(removed_blur) + len(removed_dark) + len(removed_dup)
        n_after   = len(remaining)
        total_removed  += n_removed
        total_enhanced += len(enhanced_list) if apply else len(remaining)

        class_summary[cls_dir.name] = {
            "original":   len(images),
            "removed":    n_removed,
            "remaining":  n_after,
            "blur":       len(removed_blur),
            "dark":       len(removed_dark),
            "duplicates": len(removed_dup),
            "mislabels":  len(mislabels),
        }

        # Report lines per class
        report_lines.append(
            f"\n  {cls_dir.name}"
            f"  orig={len(images)}  removed={n_removed}"
            f"  (blur={len(removed_blur)} dark={len(removed_dark)}"
            f" dup={len(removed_dup)})  remaining={n_after}"
        )

        for fp, sim in mislabels:
            mislabel_lines.append(
                f"  {cls_dir.name}/{fp.name}  similarity={sim:.4f}"
            )

    # ── SUMMARY REPORT ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  QUALITY REPORT")
    print("=" * 65)
    print(f"\n  {'Class':<38} {'Orig':>6} {'Removed':>8} "
          f"{'Blur':>6} {'Dark':>6} {'Dup':>5} {'Left':>6}")
    print("-" * 65)
    for cls, s in sorted(class_summary.items(),
                         key=lambda x: -x[1]["removed"]):
        flag = " ← LOW" if s["remaining"] < 30 else ""
        print(f"  {cls:<38} {s['original']:>6} {s['removed']:>8} "
              f"{s['blur']:>6} {s['dark']:>6} {s['duplicates']:>5} "
              f"{s['remaining']:>6}{flag}")
    print("-" * 65)
    print(f"  Total removed : {total_removed}")
    print(f"  Mislabel candidates flagged: {len(mislabel_lines)}")

    # Save text report
    rp = REPORT_DIR / "quality_report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write(f"  Safe Pharmacy — Data Quality Report  [{mode}]\n")
        f.write("=" * 65 + "\n")
        f.write(f"\n  Total images removed : {total_removed}\n")
        f.write(f"  Mislabel candidates  : {len(mislabel_lines)}\n\n")
        f.write("-" * 65 + "\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write("\n\n  AUGMENTATION RECOMMENDATION\n")
        f.write("-" * 65 + "\n")
        f.write(f"  Set in split_and_preprocess_final.py:\n")
        f.write(f"    TARGET_PER_CLASS = {TARGET_PER_CLASS_NEW}"
                f"  (was 350)\n")
        f.write(f"    MAX_COPIES       = {MAX_COPIES_NEW}"
                f"  (was 12)\n")
    print(f"\n  Report saved: {rp}")

    # Save mislabel candidates
    mp = REPORT_DIR / "mislabel_candidates.txt"
    with open(mp, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  Mislabel Candidates (review manually)\n")
        f.write("  These images scored lowest similarity to their class.\n")
        f.write("=" * 65 + "\n\n")
        if mislabel_lines:
            for line in mislabel_lines:
                f.write(line + "\n")
        else:
            f.write("  No candidates found.\n")
    print(f"  Mislabel report: {mp}")

    # Print augmentation recommendation
    print(f"\n  RECOMMENDED AUGMENTATION SETTINGS:")
    print(f"    TARGET_PER_CLASS = {TARGET_PER_CLASS_NEW}  (in split_and_preprocess_final.py)")
    print(f"    MAX_COPIES       = {MAX_COPIES_NEW}")

    if not apply:
        print(f"\n  Run with --apply to actually apply all fixes.")
    else:
        print(f"\n  Done. Now re-run split_and_preprocess_final.py")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Safe Pharmacy data quality fixer"
    )
    parser.add_argument("--apply",       action="store_true",
                        help="Actually remove and fix images (default: dry run)")
    parser.add_argument("--report-only", action="store_true",
                        help="Save report without removing images")
    args = parser.parse_args()

    run_pipeline(apply=args.apply, report_only=args.report_only)