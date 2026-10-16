# Hospital-shift-robust metastasis detection (work in progress)

> One-sentence summary, demo link and key result will be added here once they exist.
> Every number in this README will come from a file in `results/`.

**Status:** Phase 0 — environment and project structure. Target for v1: November 2026.

## Goal

Detect lymph-node metastases in 96×96 histopathology patches (Camelyon17-WILDS) and measure how well models
generalize to a hospital never seen during training. Compared approaches: a CNN baseline, stain-color handling
(HED augmentation, Macenko normalization), and frozen features from ImageNet vs. pathology foundation models (Phikon),
with per-hospital metrics, calibration and selective prediction (abstention).

## Check your environment

```bash
pip install -r requirements.txt
python -m src.check_env
```

## Development note

This project is developed with the help of Claude Code (Anthropic), used as a pair-programmer and mentor.

## Limitations, licenses, reproduction

To be completed (Phase 7).
