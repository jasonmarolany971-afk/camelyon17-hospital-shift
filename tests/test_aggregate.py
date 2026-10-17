import json

import pandas as pd
import pytest

from src.aggregate import load_runs, long_table, summarize


def fake_metrics(accuracy: float, epoch: int) -> dict:
    hospital = {"n": 10, "accuracy": accuracy, "auc": 0.9,
                "sensitivity_at_threshold": 0.5, "specificity_at_threshold": 0.95}
    split = {**hospital, "official_accuracy": accuracy, "by_hospital": {"2": hospital}}
    return {"selected_epoch": epoch, "threshold": {"value": 0.7},
            "splits": {"id_val": split, "val": split, "test": split}}


@pytest.fixture
def runs_dir(tmp_path):
    for seed, accuracy, epoch in [(0, 0.5, 7), (1, 0.8, 4), (2, 0.8, 1)]:
        folder = tmp_path / f"seed{seed}"
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps(fake_metrics(accuracy, epoch)))
        pd.DataFrame({"epoch": [0], "id_val_acc": [0.9], "val_acc": [0.8]}).to_csv(folder / "history.csv", index=False)
    (tmp_path / "seed0-subset512").mkdir()  # quick test folder: must be ignored
    return tmp_path


def test_load_runs_keeps_complete_runs_sorted_by_seed(runs_dir):
    assert list(load_runs(runs_dir)) == [0, 1, 2]


def test_summary_uses_sample_standard_deviation(runs_dir):
    summary = summarize(long_table(load_runs(runs_dir))).set_index(["split", "hospital", "metric"])
    row = summary.loc[("test", "all", "accuracy")]
    assert row["n_seeds"] == 3
    assert row["mean"] == pytest.approx(0.7)
    # Values 0.5, 0.8, 0.8: squared deviations 0.04 + 0.01 + 0.01 = 0.06, divided by n - 1 = 2.
    assert row["std"] == pytest.approx(0.06 ** 0.5 / 2 ** 0.5)
    assert row["values"] == "seed0=0.5000;seed1=0.8000;seed2=0.8000"
    assert summary.loc[("val", "all", "selected_epoch"), "values"] == "seed0=7.0000;seed1=4.0000;seed2=1.0000"
    assert ("test", "2", "accuracy") in summary.index
