"""Extract frozen features of every patch, once, with a pretrained ViT-B/16.

Usage (from the repository root):
    python -m src.features --model phikon
    python -m src.features --model dino_vitb16
    python -m src.features --model phikon --subset 512      # quick test

For each split, writes to <out-dir>/<model>/:
    <split>.npy         float16 array, one 768-d row per patch (the [CLS] token)
    <split>_index.npy   dataset index of each row, to align rows with labels and metadata
    meta.json           model, pinned revision, preprocessing, speed, code commit

A split whose file already exists is skipped, so an interrupted extraction
resumes where it stopped. Limitation of the method: features are computed
once, so no data augmentation is possible on top of them.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from src.data import MEAN, STD, get_subset, load_dataset, make_loader
from src.train import code_commit
from src.utils import get_device, save_json, synchronize

# Pinned revisions: the weights cannot change under us between two runs.
BACKBONES = {
    "phikon": {
        "hf_id": "owkin/phikon",
        "revision": "057cc0295895c2df3dd7681a89680da6015cbefe",
        "pretraining": "iBOT on 40M TCGA histology tiles",
        "license": "Owkin non-commercial license",
    },
    "dino_vitb16": {
        "hf_id": "facebook/dino-vitb16",
        "revision": "f205d5d8e640a89a2b8ef0369670dfc37cc07fc2",
        "pretraining": "DINO on ImageNet-1k (natural images)",
        "license": "Apache-2.0",
    },
}
SPLITS = ["train", "id_val", "val", "test"]
INPUT_SIZE = 224


def build_feature_transform():
    """Same preprocessing as both models' preprocessor_config.json.

    Bilinear resize of the 96x96 patch to 224x224 (the size the models were
    trained on), then ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_backbone(name: str, device: torch.device):
    from transformers import ViTModel

    spec = BACKBONES[name]
    model = ViTModel.from_pretrained(spec["hf_id"], revision=spec["revision"], add_pooling_layer=False)
    return model.eval().to(device)


@torch.no_grad()
def extract(model, loader, device, use_amp: bool) -> np.ndarray:
    chunks = []
    for x, _, _ in tqdm(loader, desc="extract", leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            hidden = model(pixel_values=x.to(device, non_blocking=True)).last_hidden_state
        chunks.append(hidden[:, 0].float().cpu().numpy().astype(np.float16))  # [CLS] token
    return np.concatenate(chunks)


def save_atomically(array: np.ndarray, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen features with a pretrained backbone.")
    parser.add_argument("--model", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset", type=int, default=None, help="Use N random patches per split (quick tests).")
    args = parser.parse_args()

    device = get_device(args.device)
    # Inference only: fp16 is faster (+14% measured on MPS) and the features are unchanged
    # (cosine similarity with fp32 >= 0.9999); they are stored in float16 anyway.
    use_amp = device.type in ("cuda", "mps")
    out_dir = args.out_dir / (args.model + (f"-subset{args.subset}" if args.subset else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_root)
    transform = build_feature_transform()
    model = load_backbone(args.model, device)
    print(f"Extracting {args.model} features on {device} (amp={use_amp}) into {out_dir}")

    timings = {}
    for split in SPLITS:
        if (out_dir / f"{split}.npy").exists():
            print(f"  {split}: already extracted, skipped")
            continue
        subset = get_subset(dataset, split, transform, args.subset, seed=0)
        loader = make_loader(subset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        start = time.time()
        features = extract(model, loader, device, use_amp)
        synchronize(device)
        seconds = time.time() - start
        save_atomically(features, out_dir / f"{split}.npy")
        save_atomically(np.asarray(subset.indices, dtype=np.int64), out_dir / f"{split}_index.npy")
        timings[split] = {"n": len(subset), "seconds": seconds, "images_per_s": len(subset) / seconds}
        print(f"  {split}: {features.shape} in {seconds / 60:.1f} min ({len(subset) / seconds:.0f} images/s)")

    import transformers

    save_json({
        "model": args.model,
        **BACKBONES[args.model],
        "feature": "[CLS] token of the last layer, float16",
        "preprocessing": f"bilinear resize 96 -> {INPUT_SIZE}, ImageNet normalization",
        "device": str(device),
        "amp": use_amp,
        "timings": timings,
        "subset": args.subset,
        "code_commit": code_commit(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }, out_dir / "meta.json")
    print(f"Done: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
