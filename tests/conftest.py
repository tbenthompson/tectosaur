import os
import pytest

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--tctquiet", action="store_true", help="hide debug logging")
    parser.addoption(
        "--save-golden-masters", action="store_true",
        help="reset golden master benchmarks"
    )
