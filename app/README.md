---
title: Metastasis Detection Across Hospitals
emoji: 🔬
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: other
short_description: Research demo, not a medical device
---

# Metastasis detection across hospitals

Classifies a 96 x 96 patch of an H&E lymph-node slide as containing breast-cancer metastasis or
not, refers its least confident patches to a pathologist, and shows a heat map of what pushes the
decision towards "tumor".

**Research tool, not a medical device. Do not use for diagnosis.**

The model (frozen Phikon ViT-B/16 + a linear layer) was trained on three hospitals of
Camelyon17-WILDS and chosen on a fourth one; its results on a fifth, never-seen hospital, their
uncertainty and their limits are in the "About this model" panel of the app and in the project
repository: https://github.com/jasonmarolany971-afk/camelyon17-hospital-shift

Everything in this Space is produced by `python -m src.export_app` in that repository
(`model/head.json`, `model/card.json`, `examples/`).

## Licenses

- Data: Camelyon17 (CC0 1.0); example patches come from the Camelyon17-WILDS release.
- Phikon weights (downloaded at start-up, pinned revision): Owkin non-commercial license. This demo
  is for non-commercial research and education only.
- Code: see the project repository.
