import pytest
import tectosaur.fmm.fmm_wrapper as fmm

@pytest.fixture(params = [2, 3])
def dim(request):
    return request.param

module = dict()
module[2] = fmm.two
module[3] = fmm.three
