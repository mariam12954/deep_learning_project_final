"""
organize_v3.py  —  Safe Pharmacy
=================================
Changes from v2:
  • Split ratio: 70 / 20 / 10  (more val data → catches overfitting earlier)
  • MIN_IMAGES_PER_CLASS kept at 20
  • Same 11-class merge map as v2 (no changes there)
  • Writes directly to  dataset/images/<class>/
    (same as before — split_and_preprocess_v3.py reads from here)
"""

import json
import re
import shutil
from pathlib import Path
from collections import defaultdict

BASE_DIR   = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "dataset"

MIN_IMAGES_PER_CLASS = 20
IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}

# ── Class descriptions (11 classes) ───────────────────────────
CLASS_INFO = {
    "eye_ear_nose_preparations":  {"description": "Eyes, ears, nose treatments.",         "note": "Avoid contamination of applicators."},
    "dermatology":                {"description": "Skin conditions — acne, eczema, etc.", "note": "Follow application instructions."},
    "diabetes_endocrine":         {"description": "Blood sugar & endocrine disorders.",   "note": "Monitor glucose regularly."},
    "vitamins_supplements":       {"description": "Vitamins and mineral supplements.",    "note": "Not a substitute for balanced diet."},
    "steroids_topicals":          {"description": "Topical anti-inflammatory steroids.",  "note": "Avoid prolonged use."},
    "cardiovascular_blood":       {"description": "Heart & blood-pressure medicines.",    "note": "Follow dosage strictly."},
    "analgesics_pain_fever":      {"description": "Pain relief and fever reduction.",     "note": "Do not exceed recommended dose."},
    "infections_immunity":        {"description": "Antibiotics, antifungals, immunology.","note": "Complete the full course."},
    "hormones_oncology":          {"description": "Hormones, oncology, anaesthetics.",    "note": "Strict medical supervision required."},
    "neuro_musculo":              {"description": "Neurology, psychiatry, musculoskeletal.","note": "Do not stop abruptly without advice."},
    "respiratory_digestive":      {"description": "Respiratory & digestive treatments.",  "note": "Take digestive meds around meals."},
}

# ── Merge map (same as v2) ─────────────────────────────────────
MERGE_MAP = {
    "eye_ear_nose_preparations": ["eye","eyeearnose","eye_allergy_inflammation","eye_irrigation","eye_local_anaesthetic","ear_preparation","nose_preparation","glaucoma_treatment","mydriatics"],
    "dermatology":               ["anti_histaminesanti_inflam","anti_acne","acne_preparations","psoriasiseczema","vitiligo_treatment","wartsanti_corn_preparations"],
    "diabetes_endocrine":        ["diabetes_care","insulins","hypo_glycaemics_antidiabetic","endocrine_system","anti_hyper_thyroidism","anti_hypo_thyroidism","growth_hormone","anabolic"],
    "vitamins_supplements":      ["vitamins_or_minerals","multivitamins","nutrition_supplements","nutrientsblood_electrolytes"],
    "steroids_topicals":         ["steroid","steroid_anti_biotic","topical_steroid","topical_steroid_anti_biotic","anti_fungal_steroid","topical_prepareation","topical_anti_biotic","gluco_corticoid","burnswounds"],
    "cardiovascular_blood":      ["cardio_vascular_system","anti_hypertensives","angina_treatment","anti_arrhythmics","anti_coagulants","haemostaticscoagulants","lipids_regulation","congestive_heart_failure","circulatory_disturbance_agent","anti_hypotension","vascularitics","varicose_veins","anaemia"],
    "analgesics_pain_fever":     ["analgesic_a_rheumatic","analgesica_rheumatic","non_narcotic_analgesic","headachefever","migraine_treatment","other_anti_rheumatics","gout_treatment"],
    "infections_immunity":       ["anti_biotics","infections","topical_anti_biotic","antiseptic","urinary_tract_antiseptic","anthelmintics","anti_dysentericamoebicparas","anti_virals","topical_anti_viral","scabies_lice","anti_fungals","anti_dandruff","immunological_system","immuno_suppresives","immunomodulator","vaccines"],
    "hormones_oncology":         ["female_sex_hormones","male_sex_horm_androgens","contraceptives","infertility_treatment","menopausalgyn_disorders","gynaecologyurinary_tract_dis","prostatic_hyperplasia","anti_galactorrhoea","male_sexual_tonics","cancer_therapy","alkylating_agent","anti_metabolites","cytostatic_anti_androgen","cytostatic_anti_oestrogen","cytostatic_elgonadtropin_analogu","monoclonal_antibodies","interferons","neutropenia","general_anaesthetic","local_anaesthetic","anti_dote","enzyme_inhibitor","plasma_substituent_expander","plasma_substituentexpander"],
    "neuro_musculo":             ["central_nervous_system","cerebral_stimulant","cns_stimulants","neurotonic","psychotic_disorders","anti_depressant","anti_epileptic_a_convulsant","anti_parkinsonism","alzheimer_treatment","adhdattentdeficithyperacd","sedatives_hypnotics","nocturnal_enuresis","mytonics","musculo_skeletal_system","skeletal_muscle_relaxant","osteoporosis_arthritis_manag","osteoporosisarthritis_manag"],
    "respiratory_digestive":     ["respiratory_system","bronchodilator","cough_expectorant_sedative","mucolytic_muco_regulator","anti_catarrhals","anti_tussive","lozenges","topical_treatment_of_the_mouth","gastro_intestinal_tract","antacids","digestive","anti_diarrhoeal","catharticlaxativepurgative","anti_emetic","anti_flatulence","anti_spasmodic","ulcer_treatment","haemorrhoids_anal_fissures","liver_disease_management"],
}

SOURCE_TO_TARGET = {src: tgt for tgt, srcs in MERGE_MAP.items() for src in srcs}


def extract_class(filename: str) -> str:
    name = re.sub(r"\.(webp|jpg|jpeg|png|jfif)$", "", filename, flags=re.IGNORECASE)
    return re.sub(r"_\d+$", "", name)


def organize_images():
    images_dir = Path(OUTPUT_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    copied, skipped = defaultdict(int), 0
    for img_file in Path(SOURCE_DIR).iterdir():
        if img_file.suffix.lower() not in IMAGE_EXTS:
            continue
        target = SOURCE_TO_TARGET.get(extract_class(img_file.name))
        if target is None:
            skipped += 1
            continue
        dest = images_dir / target
        dest.mkdir(exist_ok=True)
        shutil.copy2(img_file, dest / img_file.name)
        copied[target] += 1
    return dict(copied), skipped


def remove_small_classes(copied):
    images_dir = Path(OUTPUT_DIR) / "images"
    removed = {}
    for cls, count in list(copied.items()):
        if count < MIN_IMAGES_PER_CLASS:
            shutil.rmtree(images_dir / cls, ignore_errors=True)
            removed[cls] = count
            del copied[cls]
    return removed


def write_report(final, removed, skipped):
    text_dir = Path(OUTPUT_DIR) / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 60,
        "  Safe Pharmacy - Dataset Report  (split: 70/20/10 split)",
        "=" * 60,
        f"\nFinal classes : {len(final)}",
        f"Total images  : {sum(final.values())}",
        f"Non-medicine  : {skipped} excluded",
        f"Removed (< {MIN_IMAGES_PER_CLASS}): {len(removed)} classes\n",
        "-" * 40,
        "  FINAL CLASSES",
        "-" * 40,
    ]
    for cls, cnt in sorted(final.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:<40} {cnt:>4} images")
    text = "\n".join(lines)
    (text_dir / "class_report.txt").write_text(text, encoding="utf-8")
    print(text)
    active = {c: CLASS_INFO[c] for c in final if c in CLASS_INFO}
    (text_dir / "class_descriptions.json").write_text(
        json.dumps(active, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Report saved to: {text_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("  organize_v3.py  (11 classes, 70/20/10 split prep)")
    print("=" * 60)
    print("\n[1/3] Copying and merging images...")
    copied, skipped = organize_images()
    print(f"      {sum(copied.values())} images → {len(copied)} classes  |  {skipped} skipped")
    print(f"\n[2/3] Removing classes < {MIN_IMAGES_PER_CLASS} images...")
    removed = remove_small_classes(copied)
    print(f"      Removed: {list(removed.keys()) or 'none'}")
    print("\n[3/3] Writing report...")
    write_report(copied, removed, skipped)
    print("\n  Done.  Next → python split_and_preprocess.py")