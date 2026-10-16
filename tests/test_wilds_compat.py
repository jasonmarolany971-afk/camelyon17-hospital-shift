import torch

from src.wilds_compat import ensure_torch_scatter, scatter


def test_mean_matches_manual_group_means():
    generator = torch.Generator().manual_seed(0)
    values = torch.rand(1000, generator=generator)
    groups = torch.randint(0, 7, (1000,), generator=generator)
    groups[groups == 3] = 4  # group 3 is left empty on purpose

    result = scatter(values, groups, dim_size=8, reduce="mean")

    for g in range(8):
        members = values[groups == g]
        expected = members.mean() if members.numel() > 0 else torch.tensor(0.0)
        assert torch.allclose(result[g], expected)
    assert result[3] == 0 and result[7] == 0  # empty groups stay at 0


def test_sum_matches_manual_group_sums():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    groups = torch.tensor([0, 2, 2, 0])
    assert torch.equal(scatter(values, groups, dim_size=3, reduce="sum"), torch.tensor([5.0, 0.0, 5.0]))


def test_wilds_avg_over_groups_runs():
    ensure_torch_scatter()
    from wilds.common.utils import avg_over_groups

    values = torch.tensor([1.0, 0.0, 1.0, 1.0])
    groups = torch.tensor([0, 0, 1, 1])
    averages, counts = avg_over_groups(values, groups, n_groups=3)
    assert torch.equal(averages, torch.tensor([0.5, 1.0, 0.0]))
    assert torch.equal(counts, torch.tensor([2.0, 2.0, 0.0]))
