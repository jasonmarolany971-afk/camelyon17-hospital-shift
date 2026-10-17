"""Download Camelyon17-WILDS and lay it out exactly as the `wilds` package expects.

The official archive is hosted on CodaLab, whose TLS certificate expired on
2026-10-07, so `wilds` cannot download it. We use a public Hugging Face copy
(parquet files), pinned to a fixed commit, and rebuild the official layout:

    <data_root>/camelyon17_v1.0/
        metadata.csv
        patches/patient_XXX_node_N/patch_patient_XXX_node_N_x_X_y_Y.png
        RELEASE_v1.0.txt

PNG bytes are written unchanged (no decoding, no re-encoding). The result is
then checked against the numbers published in the WILDS paper
(Koh et al., 2021, Appendix E.2).

Usage (from the repository root):
    python -m src.prepare_data                  # download + convert + verify
    python -m src.prepare_data --verify-only    # re-run the checks only
    python -m src.prepare_data --max-files 1    # quick test on one parquet file
    python -m src.prepare_data --delete-parquet # small disks (Kaggle): ~12 GB instead of ~22 GB
"""

import argparse
import json
import os
import random
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm

HF_REPO = "wltjr1007/Camelyon17-WILDS"
# Pinned commit: the copy cannot change under us between two runs.
HF_REVISION = "d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f"
DATASET_DIR = "camelyon17_v1.0"
PARQUET_DIR = "camelyon17_hf_parquet"
# Parquet files (10.7 GB) + extracted PNG files (about the same size).
MIN_FREE_DISK_GB = 25
# With --delete-parquet, only one parquet file exists at a time.
MIN_FREE_DISK_GB_STREAMING = 13

# Official split sizes (WILDS paper, Appendix E.2).
EXPECTED_SPLIT_SIZES = {"train": 302_436, "id_val": 33_560, "val": 34_904, "test": 85_054}
TRAIN_CENTERS = {0, 3, 4}
# Values of the `split` column in metadata.csv. `wilds` only reads it for the
# training hospitals (0 = train, 1 = id_val) and overwrites it for hospitals
# 1 and 2 with `val` and `test`.
SPLIT_TRAIN, SPLIT_ID_VAL = 0, 1
METADATA_COLUMNS = ["patient", "node", "x_coord", "y_coord", "tumor", "slide", "center", "split"]


def report(ok: bool, name: str, message: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL':<4}] {name:<26} {message}")
    return ok


def patch_path(patient: int, node: int, x: int, y: int) -> str:
    """Relative path of a patch, in the format hard-coded in `Camelyon17Dataset`."""
    folder = f"patient_{patient:03d}_node_{node}"
    return f"patches/{folder}/patch_{folder}_x_{x}_y_{y}.png"


def list_parquet_files(max_files: int | None) -> list[str]:
    names = sorted(
        name
        for name in list_repo_files(HF_REPO, repo_type="dataset", revision=HF_REVISION)
        if name.endswith(".parquet")
    )
    return names if max_files is None else names[:max_files]


def download_file(name: str, parquet_dir: Path) -> Path:
    # An already-downloaded file is skipped, so an interrupted run can be resumed.
    return Path(hf_hub_download(HF_REPO, name, repo_type="dataset", revision=HF_REVISION, local_dir=parquet_dir))


def convert_file(path: Path, dataset_dir: Path, created_dirs: set) -> list[dict]:
    """Write every patch of one parquet file as a PNG file; return its metadata rows."""
    hf_split = path.name.split("-")[0]  # "train", "validation" or "test"
    parquet = pq.ParquetFile(path)
    rows = []
    with tqdm(total=parquet.metadata.num_rows, desc=f"  {path.name}", unit="patch", leave=False) as progress:
        for batch in parquet.iter_batches(batch_size=1024):
            columns = batch.to_pydict()
            for i in range(batch.num_rows):
                patient, node = columns["patient"][i], columns["node"][i]
                x, y = columns["x_coord"][i], columns["y_coord"][i]
                center = columns["center"][i]

                target = dataset_dir / patch_path(patient, node, x, y)
                if target.parent not in created_dirs:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    created_dirs.add(target.parent)
                target.write_bytes(columns["image"][i]["bytes"])

                # The HF "validation" split merges id_val (training hospitals)
                # and val (hospital 1); the hospital tells them apart.
                is_id_val = hf_split == "validation" and center in TRAIN_CENTERS
                rows.append({
                    "image_id": columns["image_id"][i],
                    "patient": f"{patient:03d}",
                    "node": node,
                    "x_coord": x,
                    "y_coord": y,
                    "tumor": columns["label"][i],
                    "slide": columns["slide"][i],
                    "center": center,
                    "split": SPLIT_ID_VAL if is_id_val else SPLIT_TRAIN,
                })
            progress.update(batch.num_rows)
    return rows


def build_metadata(rows: list[dict]) -> pd.DataFrame:
    # Sorting by image_id gives a stable row order, hence stable dataset indices.
    metadata = pd.DataFrame(rows).sort_values("image_id").set_index("image_id")
    metadata.index.name = None
    return metadata[METADATA_COLUMNS]


def write_release_file(dataset_dir: Path) -> None:
    # `wilds` treats the dataset as present only if this file exists, so it is
    # written last: an interrupted conversion is never mistaken for a complete one.
    (dataset_dir / "RELEASE_v1.0.txt").write_text(
        "Camelyon17-WILDS v1.0, rebuilt by src/prepare_data.py from the Hugging Face copy\n"
        f"{HF_REPO} at revision {HF_REVISION}.\n"
        "This is not the official CodaLab archive (its TLS certificate expired on 2026-10-07).\n"
    )


def verify(data_root: Path, n_images: int, seed: int) -> dict:
    """Check the dataset against the WILDS paper. Returns a summary dict."""
    from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

    if not (data_root / DATASET_DIR / "RELEASE_v1.0.txt").exists():
        # Without this check, wilds would suggest download=True, which cannot work.
        report(False, "dataset folder", f"no complete dataset in {data_root / DATASET_DIR}")
        return {"passed": False, "split_sizes": {}, "n_patches": 0}

    dataset = Camelyon17Dataset(root_dir=str(data_root), download=False)
    split_names = {index: name for name, index in dataset.split_dict.items()}
    splits = pd.Series(dataset.split_array).map(split_names)
    hospitals = dataset.metadata_array[:, 0].numpy()
    slides = dataset.metadata_array[:, 1].numpy()
    labels = dataset.y_array.numpy()
    meta = pd.DataFrame({"split": splits, "hospital": hospitals, "slide": slides})

    print(f"\nVerifying {dataset.data_dir}")
    results = []

    sizes = {name: int((splits == name).sum()) for name in EXPECTED_SPLIT_SIZES}
    results.append(report(sizes == EXPECTED_SPLIT_SIZES, "split sizes (paper)", str(sizes)))

    split_hospitals = meta.groupby("split").hospital.unique().apply(lambda h: sorted(h.tolist()))
    expected_hospitals = {"train": [0, 3, 4], "id_val": [0, 3, 4], "val": [1], "test": [2]}
    results.append(report(
        split_hospitals.to_dict() == expected_hospitals, "hospitals per split", str(split_hospitals.to_dict())
    ))

    slides_per_hospital = meta.groupby("hospital").slide.nunique().to_dict()
    results.append(report(
        slides_per_hospital == {h: 10 for h in range(5)}, "10 slides per hospital", str(slides_per_hospital)
    ))

    one_hospital_per_slide = bool((meta.groupby("slide").hospital.nunique() == 1).all())
    results.append(report(one_hospital_per_slide, "one hospital per slide", "each slide comes from one hospital"))

    train_slides = set(meta.slide[meta.split == "train"])
    ood_slides = set(meta.slide[meta.split.isin(["val", "test"])])
    results.append(report(
        not (train_slides & ood_slides), "no val/test slide in train", f"{len(train_slides & ood_slides)} shared"
    ))

    label_values = sorted(int(v) for v in np.unique(labels))
    results.append(report(label_values == [0, 1], "binary labels", f"values {label_values}"))

    # Two patches with the same path would silently overwrite each other.
    n_unique = len(set(dataset._input_array))
    results.append(report(n_unique == len(dataset), "unique patch paths", f"{len(dataset) - n_unique} duplicates"))

    missing = [p for p in dataset._input_array if not os.path.exists(os.path.join(dataset.data_dir, p))]
    results.append(report(not missing, "all PNG files present", f"{len(missing)} missing"))

    # Decode a random sample: right size, and alpha channel (dropped by wilds) is opaque.
    rng = random.Random(seed)
    sample = rng.sample(range(len(dataset)), k=min(n_images, len(dataset)))
    bad_size, min_alpha = 0, 255
    for index in sample:
        image = Image.open(os.path.join(dataset.data_dir, dataset._input_array[index]))
        bad_size += image.size != (96, 96)
        if image.mode == "RGBA":
            min_alpha = min(min_alpha, image.getchannel("A").getextrema()[0])
    results.append(report(bad_size == 0, "patch size 96x96", f"{bad_size}/{len(sample)} wrong size"))
    results.append(report(min_alpha >= 250, "alpha channel opaque", f"min alpha = {min_alpha} on {len(sample)} patches"))

    return {"passed": all(results), "split_sizes": sizes, "n_patches": len(dataset)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify Camelyon17-WILDS.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "~/datasets")),
        help="Folder where datasets are stored (default: $DATA_ROOT, else ~/datasets).",
    )
    parser.add_argument("--verify-only", action="store_true", help="Skip download and conversion.")
    parser.add_argument("--max-files", type=int, default=None, help="Only use the first N parquet files (testing).")
    parser.add_argument("--delete-parquet", action="store_true",
                        help="Delete each parquet file once converted (needs ~12 GB instead of ~22 GB).")
    parser.add_argument("--n-images", type=int, default=2000, help="Number of patches decoded during verification.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the verification sample.")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("results/data_provenance.json"),
        help="Where to write the provenance record (source, revision, checks).",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    dataset_dir = data_root / DATASET_DIR

    if not args.verify_only:
        data_root.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(data_root).free / 1e9
        needed_gb = MIN_FREE_DISK_GB_STREAMING if args.delete_parquet else MIN_FREE_DISK_GB
        if free_gb < needed_gb:
            print(f"Only {free_gb:.0f} GB free in {data_root}, need {needed_gb} GB.")
            return 1

        names = list_parquet_files(args.max_files)
        print(f"1/2 Downloading and converting {len(names)} parquet files from {HF_REPO}@{HF_REVISION[:8]} "
              f"into {dataset_dir} ...")
        created_dirs, rows = set(), []
        for name in tqdm(names, desc="Parquet files", unit="file"):
            path = download_file(name, data_root / PARQUET_DIR)
            rows += convert_file(path, dataset_dir, created_dirs)
            if args.delete_parquet:
                path.unlink()
        build_metadata(rows).to_csv(dataset_dir / "metadata.csv")
        write_release_file(dataset_dir)
        print("2/2 Verifying ...")

    summary = verify(data_root, args.n_images, args.seed)
    # A provenance record only makes sense for the full dataset.
    if args.max_files is None and summary["n_patches"] > 0:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(json.dumps({
            "source": f"https://huggingface.co/datasets/{HF_REPO}",
            "revision": HF_REVISION,
            "official_source": "https://worksheets.codalab.org (WILDS v1.0, unreachable: expired TLS certificate)",
            "checked_on": date.today().isoformat(),
            **summary,
        }, indent=2) + "\n")
        print(f"\nProvenance written to {args.provenance}")

    print("\nAll checks passed." if summary["passed"] else "\nSome checks failed: see the FAIL lines above.")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
