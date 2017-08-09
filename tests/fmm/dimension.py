import pytest

@pytest.fixture(params = [2, 3])
def dim(request):
    return request.param
