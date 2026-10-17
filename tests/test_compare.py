import json

import pandas as pd
import pytest

from src.compare import collect, comparison_table
from tests.test_aggregate import fake_metrics


@pytest.fixture
def runs_root(tmp_path):
    for config, accuracies in {"a": [0.5, 0.7], "b": [0.9, 0.9]}.items():
        for seed, accuracy in enumerate(accuracies):
            folder = tmp_path / config / f"seed{seed}"
            folder.mkdir(parents=True)
            (folder / "metrics.json").write_text(json.dumps(fake_metrics(accuracy, epoch=0)))
            pd.DataFrame({"epoch": [0], "id_val_acc": [0.9], "val_acc": [0.8]}).to_csv(folder / "history.csv", index=False)
    return tmp_path


def test_comparison_table_one_row_per_config_and_split(runs_root):
    table = comparison_table(collect(["a", "b"], runs_root), ["a", "b"]).set_index(["config", "split"])
    assert len(table) == 6  # 2 configurations x 3 splits
    assert table.loc[("a", "test"), "accuracy_mean"] == pytest.approx(0.6)
    assert table.loc[("a", "test"), "accuracy_std"] == pytest.approx(0.02 ** 0.5)  # n - 1 = 1
    assert table.loc[("b", "test"), "n_seeds"] == 2
