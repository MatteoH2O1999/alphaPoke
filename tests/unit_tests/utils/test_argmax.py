import pytest

from utils import argmax


def test_argmax():
    test1 = [0, 35, 678, 5]
    assert argmax(test1) == 2
    test2 = []
    with pytest.raises(ValueError):
        argmax(test2)
    test3 = [56.7]
    assert argmax(test3) == 0
    test4 = [0, 0, 0, 0, 0, 0, 0, 0, 2.3, 1, 2.29]
    assert argmax(test4) == 8
