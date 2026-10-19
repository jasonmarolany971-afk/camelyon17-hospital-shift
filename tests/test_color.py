import numpy as np
import torch
from PIL import Image

from src.color import (HEDJitter, MacenkoNormalizer, estimate_stains, od_to_rgb, reconstruction_overflows, rgb_to_od,
                       tissue_od, to_image)

HEMATOXYLIN = np.array([0.65, 0.70, 0.29])
EOSIN = np.array([0.07, 0.99, 0.11])


def unit(vectors):
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def synthetic_od(stains, seed=0, n=96 * 96):
    """OD pixels mixing two known stains, with some pure-H and pure-E pixels and some background."""
    rng = np.random.default_rng(seed)
    concentrations = rng.uniform(0.2, 1.2, size=(n, 2))
    concentrations[: n // 10, 1] = 0.0          # pure hematoxylin
    concentrations[n // 10: n // 5, 0] = 0.0     # pure eosin
    concentrations[-n // 10:] = 0.0              # white background
    return concentrations @ unit(stains)


def synthetic_image(stains, seed=0):
    return to_image(od_to_rgb(synthetic_od(stains, seed)).reshape(96, 96, 3))


def test_od_round_trip():
    rgb = np.random.default_rng(0).uniform(0.01, 1.0, size=(4, 4, 3))
    assert np.allclose(od_to_rgb(rgb_to_od(rgb)), rgb)


def test_estimate_stains_recovers_known_vectors_in_hematoxylin_eosin_order():
    od = synthetic_od([HEMATOXYLIN, EOSIN])
    tissue = np.linalg.norm(od, axis=1) >= 0.15
    found = estimate_stains(od[tissue])
    cosines = (found * unit([HEMATOXYLIN, EOSIN])).sum(axis=1)
    assert np.all(cosines > 0.999)


def test_macenko_maps_source_stains_to_reference_stains():
    reference = unit([HEMATOXYLIN, EOSIN])
    normalizer = MacenkoNormalizer(reference, max_concentrations=[1.0, 1.0])
    source = synthetic_image([[0.55, 0.80, 0.25], [0.15, 0.90, 0.30]])  # another lab's stains
    normalized, ok = normalizer.normalize(source)
    assert ok
    _, od, tissue = tissue_od(normalized)
    cosines = (estimate_stains(od[tissue]) * reference).sum(axis=1)
    assert np.all(cosines > 0.99)


def test_macenko_falls_back_on_background():
    white = Image.new("RGB", (96, 96), (250, 250, 250))
    image, ok = MacenkoNormalizer(unit([HEMATOXYLIN, EOSIN]), [1.0, 1.0]).normalize(white)
    assert not ok and image is white


def test_hed_jitter_without_noise_is_identity():
    image = synthetic_image([HEMATOXYLIN, EOSIN])
    assert np.array_equal(np.asarray(HEDJitter(sigma=0.0)(image)), np.asarray(image))


def test_hed_light_changes_colors_by_a_few_levels_and_is_seeded_by_torch():
    image = synthetic_image([HEMATOXYLIN, EOSIN])
    torch.manual_seed(0)
    first = np.asarray(HEDJitter(sigma=0.05)(image), dtype=float)
    torch.manual_seed(0)
    again = np.asarray(HEDJitter(sigma=0.05)(image), dtype=float)
    change = np.abs(first - np.asarray(image, dtype=float))
    assert np.array_equal(first, again)
    assert 0 < change.mean() < 15  # out of 255: a light change


def test_pinv_concentrations_match_least_squares():
    od = synthetic_od([HEMATOXYLIN, EOSIN])
    stains = unit([HEMATOXYLIN, EOSIN])
    from src.color import stain_concentrations
    expected = np.linalg.lstsq(stains.T, od.T, rcond=None)[0].T
    assert np.allclose(stain_concentrations(od, stains), expected)


def test_reconstruction_overflows_flags_overflow_only():
    class Overflowing:  # stands in for a patch whose reconstruction overflows exp()
        def normalize(self, image):
            return np.exp(np.array([1000.0])), True

    assert reconstruction_overflows(Overflowing(), image=None)
    normal = MacenkoNormalizer(unit([HEMATOXYLIN, EOSIN]), max_concentrations=[1.0, 1.0])
    assert not reconstruction_overflows(normal, synthetic_image([HEMATOXYLIN, EOSIN]))
