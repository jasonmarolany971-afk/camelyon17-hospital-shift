import argparse
from pathlib import Path

import pytest

from src.run_jobs import last_epoch_line, parse_job
from src.train import run_dir_for


def test_parse_job():
    assert parse_job("configs/cnn_none.yaml:2") == ("configs/cnn_none.yaml", 2)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_job("configs/cnn_none.yaml")


def test_last_epoch_line_ignores_progress_bars():
    log = (
        "Training cnn_none seed 0 on cuda\n"
        "train:  40%|####      | 3780/9451\rtrain:  80%|########  | 7560/9451\r"
        "epoch 0: loss 0.3 | val acc 0.80\n"
        "train:  10%|#         | 945/9451\r"
        "epoch 1: loss 0.2 | val acc 0.85  <- best so far\n"
        "train:   5%|          | 472/9451\r"
    )
    assert last_epoch_line(log) == "epoch 1: loss 0.2 | val acc 0.85  <- best so far"
    assert last_epoch_line("train:  5%|  | 472/9451\r") is None


def test_run_dir_for_keeps_quick_tests_apart():
    assert run_dir_for(Path("runs"), "cnn_none", 0) == Path("runs/cnn_none/seed0")
    assert run_dir_for(Path("runs"), "cnn_none", 0, subset=512) == Path("runs/cnn_none/seed0-subset512")
