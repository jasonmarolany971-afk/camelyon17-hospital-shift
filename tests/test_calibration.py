import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.calibration import apply_temperature, brier, calibrate_run, ece, fit_temperature, to_logit


def calibrated_sample(n=200_000, spread=3.0, seed=0):
    """Logits and labels drawn from the logits themselves: perfectly calibrated by construction."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, spread, n)
    return z, (rng.random(n) < expit(z)).astype(int)


def test_logit_round_trip():
    prob = np.array([0.001, 0.2, 0.5, 0.9, 0.999])
    assert np.allclose(expit(to_logit(prob)), prob)


def test_ece_is_small_when_calibrated_and_exact_when_overconfident():
    z, y = calibrated_sample()
    assert ece(expit(z), y) < 0.01
    # Always 0.99 sure, right 80% of the time: one bin, |0.99 - 0.8| = 0.19.
    y = np.array([1] * 80 + [0] * 20)
    assert ece(np.full(100, 0.99), y) == pytest.approx(0.19)


def test_fit_temperature_recovers_known_overconfidence():
    z, y = calibrated_sample()
    assert fit_temperature(expit(2.0 * z), y) == pytest.approx(2.0, rel=0.03)   # twice too confident
    assert fit_temperature(expit(0.5 * z), y) == pytest.approx(0.5, rel=0.03)   # twice too timid


def test_temperature_keeps_decisions_and_ranking():
    prob = np.random.default_rng(1).random(1000)
    scaled = apply_temperature(prob, 2.5)
    assert np.array_equal(scaled > 0.5, prob > 0.5)
    assert np.array_equal(np.argsort(scaled, kind="stable"), np.argsort(prob, kind="stable"))


def test_scaling_lowers_ece_and_brier_of_an_overconfident_model():
    z, y = calibrated_sample(seed=2)
    prob = expit(3.0 * z)
    scaled = apply_temperature(prob, fit_temperature(prob, y))
    assert ece(scaled, y) < ece(prob, y) and brier(scaled, y) < brier(prob, y)


def test_calibrate_run_fits_on_val_only(tmp_path):
    """Changing test labels must not change the temperature."""
    (tmp_path / "predictions").mkdir()
    rng = np.random.default_rng(3)
    for split, hospitals in [("id_val", [0, 3]), ("val", [1]), ("test", [2])]:
        z, y = calibrated_sample(n=4000, seed=len(split))
        pd.DataFrame({"index": range(len(z)), "y": y, "prob": expit(2.0 * z).astype("float32"),
                      "hospital": rng.choice(hospitals, len(z)), "slide": 0}).to_parquet(
            tmp_path / "predictions" / f"{split}.parquet", index=False)
    first, _ = calibrate_run(tmp_path)
    test = pd.read_parquet(tmp_path / "predictions" / "test.parquet")
    test.assign(y=1 - test.y).to_parquet(tmp_path / "predictions" / "test.parquet", index=False)
    second, _ = calibrate_run(tmp_path)
    assert first["temperature"] == second["temperature"]
    assert set(first["splits"]["id_val"]) == {"all", "0", "3"}
