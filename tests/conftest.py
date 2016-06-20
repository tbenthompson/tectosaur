import pytest

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption(
        "--save-golden-masters", action="store_true",
        help="reset golden master benchmarks"
    )
