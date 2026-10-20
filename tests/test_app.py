"""The demo app is self-contained (it does not import src/): check that it matches the project's code."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.features import build_feature_transform


def load_app():
    spec = importlib.util.spec_from_file_location("demo_app", Path("app/app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_patch(size=(96, 96), seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))


def test_preprocessing_is_identical_to_feature_extraction():
    app = load_app()
    patch = random_patch()
    assert torch.allclose(app.preprocess(patch), build_feature_transform()(patch).unsqueeze(0), atol=1e-6)


def test_patches_of_the_training_size_are_untouched_and_others_are_cropped():
    app = load_app()
    patch = random_patch()
    same, note = app.to_patch(patch)
    assert note == "" and np.array_equal(np.asarray(same), np.asarray(patch))
    other, note = app.to_patch(random_patch(size=(300, 200)))
    assert other.size == (96, 96) and "center-cropped" in note


def test_overlay_keeps_the_image_when_the_map_is_empty():
    app = load_app()
    patch = random_patch()
    shown = app.overlay(patch, np.zeros((14, 14), dtype=np.float32))
    expected = np.asarray(patch.resize((app.DISPLAY, app.DISPLAY), Image.BILINEAR))
    third = app.DISPLAY // 3
    inside = (slice(third + 4, 2 * third - 4), slice(third + 4, 2 * third - 4))  # away from the drawn square
    assert shown.size == (app.DISPLAY, app.DISPLAY)
    assert np.array_equal(np.asarray(shown)[inside], expected[inside])
