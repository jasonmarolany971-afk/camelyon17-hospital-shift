"""Run the official WILDS evaluator without compiling torch-scatter.

`wilds` 2.0.0 imports `torch_scatter` in a single place,
`wilds.common.utils.avg_over_groups`, which averages a metric per group
(for Camelyon17: per slide) with:

    torch_scatter.scatter(src=v, index=g, dim_size=n_groups, reduce="mean")

torch-scatter is a compiled extension with no prebuilt wheel for recent
PyTorch versions on macOS. PyTorch has offered the same operation natively
since version 1.12 (`Tensor.scatter_reduce`), so when the real package is
missing we register a small stand-in module implementing that call only.
If torch-scatter is installed, the real package is used.

Usage: call `ensure_torch_scatter()` once before `dataset.eval(...)`.
"""

import importlib.util
import sys
import types

import torch


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    """Subset of `torch_scatter.scatter` used by wilds: 1-D tensors, sum or mean."""
    if out is not None or src.dim() != 1 or dim not in (0, -1):
        raise NotImplementedError("Only the 1-D call made by wilds is supported.")
    if reduce not in ("sum", "mean"):
        raise NotImplementedError(f"reduce={reduce!r} is not supported.")
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    result = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    # include_self=False: the initial zeros are not counted in the mean, and
    # groups that receive no element stay at 0, as in torch_scatter.
    return result.scatter_reduce(0, index, src, reduce=reduce, include_self=False)


def ensure_torch_scatter() -> bool:
    """Make `import torch_scatter` work. Returns True if the stand-in is used."""
    if "torch_scatter" in sys.modules or importlib.util.find_spec("torch_scatter") is not None:
        return False
    module = types.ModuleType("torch_scatter")
    module.__doc__ = "Minimal stand-in for torch_scatter, see src/wilds_compat.py."
    module.scatter = scatter
    sys.modules["torch_scatter"] = module
    return True
