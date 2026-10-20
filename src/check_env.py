"""Check that this machine can run the project.

Run from the repository root:
    python -m src.check_env
    python -m src.check_env --data-root /path/to/datasets

Checks the Python version, the required packages, the accelerator
(CUDA GPU, Apple MPS or CPU) and the free disk space where the dataset
will be stored. Exits with code 1 if a blocking requirement is missing,
so it can also be used at the top of a Colab/Kaggle notebook.
"""

import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
# Camelyon17-WILDS: 10.7 GB archive + 15 GB once extracted (WILDS README).
# Both exist at the same time during extraction, hence the margin.
MIN_FREE_DISK_GB = 30
# Import names (scikit-learn is imported as "sklearn", pyyaml as "yaml").
PACKAGES = ["torch", "torchvision", "wilds", "numpy", "pandas", "sklearn", "scipy", "matplotlib", "yaml"]


def report(level: str, name: str, message: str) -> None:
    print(f"[{level:<4}] {name:<12} {message}")


def check_python() -> bool:
    ok = sys.version_info >= MIN_PYTHON
    report("OK" if ok else "FAIL", "python", f"{platform.python_version()} ({sys.executable})")
    return ok


def check_memory() -> None:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        report("INFO", "memory", f"{total / 2**30:.0f} GiB RAM")
    except (ValueError, OSError, AttributeError):
        report("INFO", "memory", "unknown")


def check_packages() -> bool:
    all_ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            report("OK", name, getattr(module, "__version__", "installed"))
        except ImportError:
            report("FAIL", name, "not installed")
            all_ok = False
    return all_ok


def check_accelerator() -> bool:
    try:
        import torch
    except ImportError:
        report("FAIL", "accelerator", "torch is missing, cannot check")
        return False

    if torch.cuda.is_available():
        device, name = torch.device("cuda"), torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device, name = torch.device("mps"), "Apple GPU via Metal (MPS)"
    else:
        device, name = torch.device("cpu"), "no GPU found, training will be slow"

    # Run the same matrix product on the CPU and on the device: if the
    # backend works, both results agree up to float32 rounding errors.
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(512, 512, generator=generator)
    b = torch.randn(512, 512, generator=generator)
    expected = a @ b
    got = (a.to(device) @ b.to(device)).cpu()
    max_diff = (expected - got).abs().max().item()
    ok = torch.allclose(expected, got, rtol=1e-3, atol=1e-3)

    level = "FAIL" if not ok else ("WARN" if device.type == "cpu" else "OK")
    report(level, "accelerator", f"{device.type}: {name} (max |cpu - device| = {max_diff:.1e})")
    return ok


def check_disk(data_root: Path) -> bool:
    # The folder may not exist yet: measure the closest existing parent.
    probe = data_root
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    ok = free_gb >= MIN_FREE_DISK_GB
    report("OK" if ok else "FAIL", "disk", f"{free_gb:.0f} GB free for {data_root} (need >= {MIN_FREE_DISK_GB} GB)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that this machine can run the project.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "~/datasets")),
        help="Folder where datasets are stored (default: $DATA_ROOT, else ~/datasets).",
    )
    args = parser.parse_args()

    print(f"Platform: {platform.platform()} ({platform.machine()})")
    check_memory()
    results = [
        check_python(),
        check_packages(),
        check_accelerator(),
        check_disk(args.data_root.expanduser()),
    ]
    if all(results):
        print("\nAll checks passed.")
        return 0
    print("\nSome checks failed: see the FAIL lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
