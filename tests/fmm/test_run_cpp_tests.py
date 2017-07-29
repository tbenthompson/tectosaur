import sys

def test_run_cpp_tests():
    from cppimport import cppimport
    test_main = cppimport('test_main')
    test_main.run_tests(sys.argv)
