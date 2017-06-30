import pytest
import numpy as np
from functools import wraps

try:
    slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
    )
except AttributeError as e:
    def slow(ob):
        return ob


def golden_master(test_fnc):
    try:
        save = pytest.config.getoption("--save-golden-masters")
    except AttributeError as e:
        save = False
    @wraps(test_fnc)
    def wrapper():
        result = test_fnc()
        test_name = test_fnc.__name__
        filename = 'tests/golden_masters/' + test_name + '.npy'
        if save:
            np.save(filename, result)
        correct = np.load(filename)
        np.testing.assert_almost_equal(result, correct, 6)
    return wrapper

