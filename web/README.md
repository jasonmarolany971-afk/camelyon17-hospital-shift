---
title: Metastasis Detection Across Hospitals
emoji: 🔬
colorFrom: purple
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Research demo, runs in your browser
---

# Metastasis detection across hospitals (in-browser demo)

Classifies a 96 x 96 patch of an H&E lymph-node slide as containing breast-cancer metastasis or not,
refers its least confident patches to a pathologist, and shows a heat map of what pushes the decision
towards "tumor". The model (DenseNet-121 with HED stain augmentation, ONNX) runs entirely in the
browser: no image leaves the visitor's computer.

**Research tool, not a medical device. Do not use for diagnosis.**

Every file here is produced by `python -m src.export_web_demo` in the project repository, where the
model choice, its results and their limits are documented:
https://github.com/jasonmarolany971-afk/camelyon17-hospital-shift

Licenses: code and weights MIT; data Camelyon17 (CC0 1.0).
