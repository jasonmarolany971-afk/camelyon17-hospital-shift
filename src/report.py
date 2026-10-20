"""Render README.md and MODEL_CARD.md from their templates, with every number read from results/.

Usage (from the repository root):
    python -m src.report           # writes README.md and MODEL_CARD.md
    python -m src.report --check   # exit 1 if either is not what the templates and results/ give

The templates (docs/*.template.md) contain the text; each number is a tag such as
{{acc_test_cnn_macenko}}, filled here from the files of results/ (and app/model/card.json for
the demo model). A test runs --check, so a README number can neither be typed by hand nor
become stale.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.reproduce import TOLERANCE

COVERAGE, RISK_TARGET = 0.8, 0.02  # the operating points of results/tables/abstention_cutoffs.csv shown
CONFIGS = ["cnn_none", "cnn_hed", "cnn_macenko", "probe_dino_vitb16", "probe_phikon"]
NAMES = {
    "cnn_none": "CNN from scratch",
    "cnn_hed": "CNN + HED stain augmentation",
    "cnn_macenko": "CNN + Macenko stain normalization",
    "probe_dino_vitb16": "Frozen DINO ViT-B/16 (natural images) + linear layer",
    "probe_phikon": "Frozen Phikon ViT-B/16 (histology) + linear layer",
}
DOCUMENTS = {"README.md": "docs/README.template.md", "MODEL_CARD.md": "docs/MODEL_CARD.template.md"}
TAG = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def signed(x: float) -> str:
    return f"{100 * x:+.1f}"


def collect(results: Path, app_card: Path) -> dict[str, str]:
    tables = results / "tables"
    v: dict[str, str] = {}

    # Data
    patches = pd.read_csv(tables / "patches_by_hospital_split.csv")
    slides = pd.read_csv(tables / "slides.csv", dtype={"patient": str})
    v["n_patches"] = f"{patches.n_patches.sum():,}"
    for split in ("train", "id_val", "val", "test"):
        v[f"n_{split}"] = f"{patches[patches.split == split].n_patches.sum():,}"
    test_patches = patches[patches.split == "test"]
    v["tumor_share_test"] = pct(test_patches.n_tumor.sum() / test_patches.n_patches.sum(), 0)
    v["n_slides"], v["n_patients"] = str(len(slides)), str(slides.patient.nunique())
    for split, hospital in (("val", 1), ("test", 2)):
        v[f"n_patients_{split}"] = str(slides[slides.hospital == hospital].patient.nunique())
    test_slides = slides[slides.hospital == 2]
    top = test_slides.loc[test_slides.n_tumor.idxmax()]
    v["top_test_slide"] = str(int(top.slide))
    v["top_test_slide_tumor_share"] = pct(top.n_tumor / test_slides.n_tumor.sum(), 0)
    v["top_test_slide_patch_share"] = pct(top.n_patches / test_slides.n_patches.sum(), 0)

    # Accuracy, AUC and 95% intervals over patients
    comparison = pd.read_csv(tables / "comparison.csv").set_index(["config", "split"])
    bootstrap = pd.read_csv(tables / "bootstrap.csv")
    v["n_replicates"] = f"{int(bootstrap.n_replicates.iloc[0]):,}"
    bootstrap = bootstrap.set_index(["config", "split", "metric"])
    v["n_seeds"] = str(int(comparison.n_seeds.min()))
    for c in CONFIGS:
        for split in ("id_val", "val", "test"):
            row = comparison.loc[(c, split)]
            v[f"acc_{split}_{c}"] = pct(row.accuracy_mean)
            v[f"accpm_{split}_{c}"] = f"{pct(row.accuracy_mean)} ± {pct(row.accuracy_std)}"
            v[f"auc_{split}_{c}"] = f"{row.auc_mean:.3f}"
            v[f"called_{split}_{c}"] = pct(row.predicted_tumor_share_mean)
        for split in ("val", "test"):
            ci = bootstrap.loc[(c, split, "accuracy")]
            v[f"ci_{split}_{c}"] = f"[{pct(ci.ci_low)}, {pct(ci.ci_high)}]"
    differences = pd.read_csv(tables / "bootstrap_differences.csv")

    def difference(a: str, b: str, split: str = "test") -> str:
        """a - b in accuracy points, with its 95% interval over patients."""
        rows = differences[(differences.split == split) & (differences.metric == "accuracy")]
        row = rows[(rows.config_a == a) & (rows.config_b == b)]
        sign = 1
        if row.empty:
            row, sign = rows[(rows.config_a == b) & (rows.config_b == a)], -1
        row = row.iloc[0]
        low, high = sorted((sign * row.ci_low, sign * row.ci_high))
        return f"{signed(sign * row.difference)} [{signed(low)}, {signed(high)}]"

    v["diff_test_macenko_hed"] = difference("cnn_macenko", "cnn_hed")
    v["diff_test_macenko_phikon"] = difference("cnn_macenko", "probe_phikon")
    v["diff_test_macenko_dino"] = difference("cnn_macenko", "probe_dino_vitb16")
    v["diff_test_phikon_dino"] = difference("probe_phikon", "probe_dino_vitb16")
    v["diff_val_phikon_macenko"] = difference("probe_phikon", "cnn_macenko", "val")
    v["diff_val_hed_macenko"] = difference("cnn_hed", "cnn_macenko", "val")
    v["main_table"] = "\n".join(
        ["| Method | Training hospitals (id_val) | Hospital 1 (val, unseen) | Hospital 2 (test, unseen) "
         "| 95% interval over test patients | Test AUC |", "|---|---|---|---|---|---|"]
        + [f"| {NAMES[c]} | {v[f'accpm_id_val_{c}']} | {v[f'accpm_val_{c}']} | **{v[f'accpm_test_{c}']}** "
           f"| {v[f'ci_test_{c}']} | {v[f'auc_test_{c}']} |" for c in CONFIGS])
    v["key_table"] = "\n".join(
        ["| Method | Accuracy on the unseen test hospital | 95% interval |", "|---|---|---|"]
        + [f"| {NAMES[c]} | **{v[f'acc_test_{c}']}%** | {v[f'ci_test_{c}']} |" for c in CONFIGS])

    reference = json.loads((tables / "wilds_paper_reference.json").read_text())
    v["wilds_test"] = f"{pct(reference['test_accuracy_mean'])} ± {pct(reference['test_accuracy_std'])}"
    cnn_summaries = list((results / "runs").glob("cnn_*/seed*/train_summary.json"))
    minutes = sum(json.loads(p.read_text())["train_minutes"] for p in cnn_summaries)
    v["cnn_gpu_hours"] = f"{minutes / 60:.0f}"
    v["n_cnn_runs"] = str(len(cnn_summaries))

    # Calibration
    calibration = pd.read_csv(tables / "calibration.csv")
    calibration = calibration[calibration.hospital == "all"].set_index(["config", "split", "metric"])
    for c in CONFIGS:
        v[f"temperature_{c}"] = f"{calibration.loc[(c, 'val', 'temperature'), 'mean']:.2f}"
        for split in ("id_val", "test"):
            for kind in ("raw", "scaled"):
                v[f"ece_{kind}_{split}_{c}"] = f"{calibration.loc[(c, split, f'ece_{kind}'), 'mean']:.3f}"
    v["calibration_table"] = "\n".join(
        ["| Method | Temperature (fitted on val) | ECE, test hospital: before → after |", "|---|---|---|"]
        + [f"| {NAMES[c]} | {v[f'temperature_{c}']} | {v[f'ece_raw_test_{c}']} → {v[f'ece_scaled_test_{c}']} |"
           for c in CONFIGS])

    # Abstention
    curves = pd.read_csv(tables / "risk_coverage.csv").set_index(["config", "split"])
    cutoffs = pd.read_csv(tables / "abstention_cutoffs.csv").set_index(["config", "rule", "target", "split"])
    for c in CONFIGS:
        v[f"risk_full_test_{c}"] = pct(curves.loc[(c, "test"), "risk_full_coverage_mean"])
        v[f"eaurc_test_{c}"] = f"{curves.loc[(c, 'test'), 'e_aurc_mean']:.3f}"
        row = cutoffs.loc[(c, "coverage", COVERAGE, "test")]
        v[f"cov80_answered_test_{c}"] = pct(row.coverage_mean, 0)
        v[f"cov80_risk_test_{c}"] = pct(row.risk_mean)
        v[f"risk02_test_{c}"] = pct(cutoffs.loc[(c, "risk", RISK_TARGET, "test"), "risk_mean"])
    v["referral_share"], v["risk_target"] = pct(1 - COVERAGE, 0), pct(RISK_TARGET, 0)
    v["tolerance"] = f"{TOLERANCE:.0e}".replace("e-0", "e-")  # 1e-9
    v["abstention_table"] = "\n".join(
        ["| Method | Error, all patches | Patches answered with referral | Error on answered patches |",
         "|---|---|---|---|"]
        + [f"| {NAMES[c]} | {v[f'risk_full_test_{c}']}% | {v[f'cov80_answered_test_{c}']}% "
           f"| {v[f'cov80_risk_test_{c}']}% |" for c in CONFIGS])

    # Errors and Grad-CAM (seed 0)
    overlap = pd.read_csv(tables / "error_overlap.csv").set_index(["split", "group"])
    shared = overlap.loc[("test", "all_methods_wrong")]
    v["shared_errors_test"], v["shared_fn_test"] = f"{shared.errors:,}", f"{shared.false_negatives:,}"
    by_slide = pd.read_csv(tables / "shared_errors_by_slide.csv")
    test_by_slide = by_slide[by_slide.split == "test"].nlargest(2, "shared_false_negatives")
    v["shared_fn_top_slides"] = " and ".join(f"{int(r.slide)} ({int(r.shared_false_negatives)})"
                                            for _, r in test_by_slide.iterrows())
    center = pd.read_csv(tables / "gradcam_center.csv").dropna(subset=["center_share_mean"])
    v["uniform_share"] = pct(center.uniform_share.iloc[0], 0)
    v["gradcam_center_min"] = pct(center.center_share_mean.min(), 0)
    v["gradcam_center_max"] = pct(center.center_share_mean.max(), 0)
    macenko_checks = json.loads((tables / "macenko_checks.json").read_text())
    v["macenko_overflow_rate"] = pct(macenko_checks["train"]["overflow_rate"], 2)

    # Demo model (app/model/card.json, written by src/export_app.py)
    card = json.loads(app_card.read_text())
    v["app_seed"] = str(card["selection"]["chosen_seed"])
    by_seed = card["selection"]["val_accuracy_by_seed"]
    v["app_val_by_seed"] = ", ".join(f"seed {s}: {pct(a, 2)}%" for s, a in by_seed.items())
    for split in ("id_val", "val", "test"):
        v[f"app_acc_{split}"] = pct(card["this_run"][split]["accuracy"])
        v[f"app_auc_{split}"] = f"{card['this_run'][split]['auc']:.3f}"
    v["app_temperature"] = f"{card['temperature']['value']:.2f}"
    before, after = card["temperature"]["ece_test_before_after"]
    v["app_ece_test_before"], v["app_ece_test_after"] = f"{before:.3f}", f"{after:.3f}"
    for split in ("val", "test"):
        referral = card["referral"][split]
        v[f"app_answered_{split}"] = pct(referral["answered"], 0)
        v[f"app_error_answered_{split}"] = pct(referral["error_when_answered"])
        v[f"app_error_all_{split}"] = pct(referral["error_without_referral"])
    app_checks = json.loads((tables / "app_checks.json").read_text())
    v["app_max_gap"] = f"{app_checks['same_model']['max_abs_probability_gap']:.4f}"
    v["app_same_decisions"] = (f"{pct(app_checks['same_model']['same_decision_share'], 0)}% of decisions identical on "
                               f"{app_checks['same_model']['n_val_patches']} validation patches")
    randomization = app_checks["heat_map_randomization"]["spearman_original_vs_random_last_block_mean"]
    v["app_randomization"] = f"{randomization:.2f}"
    return v


def render(template: Path, values: dict[str, str]) -> str:
    text = template.read_text()
    missing = sorted({tag for tag in TAG.findall(text) if tag not in values})
    if missing:
        raise KeyError(f"{template}: no value for {', '.join(missing)}")
    return TAG.sub(lambda match: values[match.group(1)], text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render README.md and MODEL_CARD.md from results/.")
    parser.add_argument("--check", action="store_true", help="Only check that both files are up to date.")
    args = parser.parse_args()

    values = collect(Path("results"), Path("app/model/card.json"))
    stale = []
    for document, template in DOCUMENTS.items():
        text = render(Path(template), values)
        if args.check:
            if not Path(document).exists() or Path(document).read_text() != text:
                stale.append(document)
        else:
            Path(document).write_text(text)
            print(f"-> {document}")
    if stale:
        print(f"Not up to date with results/: {', '.join(stale)}. Run: python -m src.report")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
