from src.reproduce import numbers_match


def test_rounding_of_the_last_digit_is_tolerated():
    assert numbers_match({"auc": 0.9862365866116289, "n": 85054}, {"auc": 0.9862365866116288, "n": 85054})


def test_real_differences_are_caught():
    assert not numbers_match({"auc": 0.986}, {"auc": 0.987})
    assert not numbers_match({"auc": 0.986}, {"accuracy": 0.986})          # other keys
    assert not numbers_match({"run": "cnn_hed/seed0"}, {"run": "cnn_hed/seed1"})
    assert not numbers_match([0.1, 0.2], [0.1])
