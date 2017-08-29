import sys

def test_run_cpp_tests():
    from cppimport import cppimport
    cppimport('test_main').run_tests()
