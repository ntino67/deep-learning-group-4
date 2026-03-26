import pytest


def init(x):
    return x + 1


def test_answer():
    assert init(3) == 4
