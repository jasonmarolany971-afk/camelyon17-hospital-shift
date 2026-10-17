import torch
import torch.nn as nn

from src.probe import batches, fit_standardizer, train_epoch
from src.utils import set_seed


def test_standardizer_uses_given_split_only():
    train = torch.tensor([[0.0, 10.0], [2.0, 30.0]])
    mean, std = fit_standardizer(train)
    assert torch.equal(mean, torch.tensor([1.0, 20.0]))
    other = torch.tensor([[1.0, 20.0]])
    assert torch.equal((other - mean) / std, torch.zeros(1, 2))  # val/test reuse the train statistics


def test_batches_cover_every_row_once():
    seen = torch.cat(list(batches(10, 3, torch.Generator().manual_seed(0))))
    assert sorted(seen.tolist()) == list(range(10))
    assert [len(b) for b in batches(10, 3, torch.Generator().manual_seed(0))] == [3, 3, 3, 1]


def train_probe(seed: int, x, y) -> dict:
    set_seed(seed)
    model = nn.Linear(x.shape[1], 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    generator = torch.Generator()
    for epoch in range(3):
        generator.manual_seed(seed * 1000 + epoch)
        stats = train_epoch(model, optimizer, x, y, batch_size=32, generator=generator)
    return {"state": model.state_dict(), **stats}


def test_training_learns_and_depends_only_on_the_seed():
    x = torch.randn(512, 8, generator=torch.Generator().manual_seed(1))
    y = (x[:, 0] > 0).long()  # linearly separable
    first, again, other = train_probe(0, x, y), train_probe(0, x, y), train_probe(1, x, y)
    assert first["train_acc"] > 0.9
    assert all(torch.equal(first["state"][k], again["state"][k]) for k in first["state"])  # bit for bit
    assert not torch.equal(first["state"]["weight"], other["state"]["weight"])
