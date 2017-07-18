import pytest
import numpy as np
from functools import wraps

from tectosaur.kernels import elastic_kernels

try:
    slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
    )
except AttributeError as e:
    def slow(ob):
        return ob


def golden_master(digits = 6):
    def decorator(test_fnc):
        try:
            save = pytest.config.getoption("--save-golden-masters")
        except AttributeError as e:
            save = False
        @wraps(test_fnc)
        def wrapper(request, *args, **kwargs):
            result = test_fnc(request, *args, **kwargs)
            test_name = request.node.name
            filename = 'tests/golden_masters/' + test_name + '.npy'
            if save:
                np.save(filename, result)
            correct = np.load(filename)
            np.testing.assert_almost_equal(result, correct, digits)
        return wrapper
    return decorator

@pytest.fixture(params = [name for name,K in elastic_kernels.items()])
def kernel(request):
    return request.param
