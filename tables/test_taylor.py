def run_tests():
    import cppimport
    test_taylor = cppimport.imp("taylor_tests")
    test_taylor.run_tests([])

if __name__ == '__main__':
    run_tests()
