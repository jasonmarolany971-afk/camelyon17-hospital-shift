"""Download the trained CNN weights of seed 0, published as a GitHub release (needed for Grad-CAM).

Usage (from the repository root):
    python -m src.download_weights

Saves artifacts/runs/<name>/seed0/best.pt for the three CNNs (about 28 MB each) and checks each
file's SHA-256 against the value recorded here. The weights were trained by this project from
scratch on Camelyon17-WILDS (CC0 data) and are released under the MIT license, like the code.
After this, python -m src.reproduce also regenerates the Grad-CAM figure.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/jasonmarolany971-afk/camelyon17-hospital-shift/releases/download/weights-seed0"
WEIGHTS = {  # configuration -> (released file, SHA-256)
    "cnn_none": ("cnn_none_seed0.pt", "e6a5bfbc27b5835caf42ce8bf6bf63084eb9e2246392b60259cb17d37f2ba202"),
    "cnn_hed": ("cnn_hed_seed0.pt", "03677ffcff21a1e2cf0c88ff6683491c4015510c02af16c0bdbc48be2069a2ee"),
    "cnn_macenko": ("cnn_macenko_seed0.pt", "d56ceaacd62000ae431350d494d85da4e3f62d4765aa493b216704de4f2e212b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for name, (file, expected) in WEIGHTS.items():
        target = Path("artifacts/runs") / name / "seed0" / "best.pt"
        if target.exists() and sha256(target) == expected:
            print(f"{name}: already there")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".download")
        print(f"{name}: downloading {file}")
        urllib.request.urlretrieve(f"{RELEASE_URL}/{file}", tmp)
        if sha256(tmp) != expected:
            tmp.unlink()
            print(f"{name}: SHA-256 mismatch, file discarded")
            return 1
        tmp.replace(target)
    print("Weights in artifacts/runs/<name>/seed0/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
