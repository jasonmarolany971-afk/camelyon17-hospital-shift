import pytest

from src.color import HEDJitter, MacenkoNormalizer
from src.data import build_transform
from src.utils import load_config


def color_steps(transform):
    return [type(step) for step in transform.transforms if isinstance(step, (HEDJitter, MacenkoNormalizer))]


@pytest.mark.parametrize("name, train_steps, eval_steps", [
    ("cnn_none", [], []),
    ("cnn_hed", [HEDJitter], []),                                   # augmentation: training only
    ("cnn_macenko", [MacenkoNormalizer], [MacenkoNormalizer]),      # normalization: everywhere
])
def test_color_step_by_config(name, train_steps, eval_steps):
    config = load_config(f"configs/{name}.yaml")
    assert color_steps(build_transform(config, train=True)) == train_steps
    assert color_steps(build_transform(config, train=False)) == eval_steps


def test_unknown_color_method_is_rejected():
    with pytest.raises(ValueError):
        build_transform({"color": "sepia"}, train=True)
